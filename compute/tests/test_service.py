import copy
from pathlib import Path

import pytest

from kernel_compute.common import failure_result
from kernel_compute.models import JobHandle, Observation
from kernel_compute.providers.base import AmbiguousSubmission, ProviderError
from kernel_compute.service import ExecutionService
from kernel_compute.store import RunStore
from kernel_compute.worker.measure import summarize

EXAMPLE = Path(__file__).parents[1] / "examples/model-new/task.toml"


class FakeProvider:
    id = "fake"

    def __init__(self):
        self.jobs = {}
        self.submissions = 0
        self.lose_ack = False
        self.fetch_error = False
        self.hidden = False
        self.fail_preflight = False

    def snapshot(self):
        return {"test": True}

    def preflight(self, request):
        if self.fail_preflight:
            raise ProviderError("preflight rejected")

    def submit(self, request, bundle):
        self.submissions += 1
        handle = JobHandle(
            provider=self.id,
            run_id=request["run_id"],
            native_id=str(self.submissions),
            cluster="fake",
            submitted_at=request["created_at"],
            artifact_locator="fake://artifact",
        )
        self.jobs[request["run_id"]] = {
            "handle": handle,
            "observation": Observation(state="queued"),
            "result": None,
        }
        if self.lose_ack:
            raise AmbiguousSubmission("lost acknowledgement")
        return handle

    def observe(self, request):
        if self.hidden:
            return None, Observation(state="unknown", reason="accounting delay")
        job = self.jobs[request["run_id"]]
        return job["handle"], job["observation"]

    def fetch(self, request, destination):
        if self.fetch_error:
            raise ProviderError("download interrupted")
        return copy.deepcopy(self.jobs[request["run_id"]]["result"])

    def cancel(self, request):
        self.jobs[request["run_id"]]["observation"] = Observation(
            state="cancelled", raw_state="CANCELLED"
        )
        return {"requested": True}


@pytest.fixture
def rig(tmp_path):
    provider = FakeProvider()
    service = ExecutionService(RunStore(tmp_path), lambda *_: provider)
    batch = service.prepare([EXAMPLE], provider, "a" * 64)
    return service, provider, batch


def success(request):
    result = failure_result(request, "succeeded", "unused", "unused")
    result["execution"]["error"] = None
    result["validation"]["status"] = "passed"
    result["latency"] = {**summarize([1.0] * 100), **request["task"]["measurement"]}
    return result


def test_lost_ack_recovers_after_new_service_without_resubmit(rig):
    service, provider, batch = rig
    provider.lose_ack = True
    provider.hidden = True
    response = service.submit_batch(batch["batch_id"])
    assert response["runs"][0]["state"] == "submitting"
    restarted = ExecutionService(RunStore(service.store.root), lambda *_: provider)
    restarted.submit_batch(batch["batch_id"])
    assert provider.submissions == 1
    provider.hidden = False
    response = restarted.submit_batch(batch["batch_id"])
    assert response["runs"][0]["handle"]["native_id"] == "1"
    assert provider.submissions == 1


def test_missing_result_never_becomes_success(rig, monkeypatch):
    service, provider, batch = rig
    service.submit_batch(batch["batch_id"])
    run_id = batch["run_ids"][0]
    provider.jobs[run_id]["observation"] = Observation(state="succeeded", raw_state="COMPLETED")
    monkeypatch.setattr("kernel_compute.service.time.time", lambda: 1000)
    assert not service.status(run_id)["result_available"]
    monkeypatch.setattr("kernel_compute.service.time.time", lambda: 1031)
    result = service.result(run_id)
    assert result["execution"]["status"] == "failed"
    assert result["execution"]["error"]["reason"] == "missing_result"


def test_interrupted_download_retries_retrieval_only(rig):
    service, provider, batch = rig
    service.submit_batch(batch["batch_id"])
    run_id = batch["run_ids"][0]
    request = service.store.get(run_id)["request"]
    provider.jobs[run_id].update(
        observation=Observation(state="succeeded"), result=success(request)
    )
    provider.fetch_error = True
    assert not service.status(run_id)["result_available"]
    provider.fetch_error = False
    assert service.result(run_id)["execution"]["status"] == "succeeded"
    assert service.result(run_id)["run_id"] == run_id
    assert provider.submissions == 1


@pytest.mark.parametrize(
    "corruption",
    ["identity", "nan", "missing_group", "skipped_validation", "units", "negative_memory"],
)
def test_corrupt_result_not_success(rig, corruption):
    service, provider, batch = rig
    service.submit_batch(batch["batch_id"])
    run_id = batch["run_ids"][0]
    result = success(service.store.get(run_id)["request"])
    if corruption == "identity":
        result["run_id"] = "another-run"
    elif corruption == "nan":
        result["latency"]["median"] = float("nan")
    elif corruption == "missing_group":
        del result["environment"]
    elif corruption == "skipped_validation":
        result["validation"]["status"] = "not_requested"
    elif corruption == "units":
        result["latency"]["units"] = "ms"
    else:
        result["memory"] = {
            "units": "bytes",
            "scope": "pytorch_allocator_including_resident_inputs_and_model",
            "peak_allocated": -1,
        }
    provider.jobs[run_id].update(observation=Observation(state="succeeded"), result=result)
    assert service.result(run_id)["execution"]["error"]["reason"] == "invalid_result"


def test_cancel_race_retains_scheduler_outcome(rig):
    service, provider, batch = rig
    service.submit_batch(batch["batch_id"])
    run_id = batch["run_ids"][0]
    service.cancel(run_id)
    provider.jobs[run_id]["result"] = success(service.store.get(run_id)["request"])
    result = service.result(run_id)
    assert result["execution"]["status"] == "cancelled"
    assert result["latency"] is None


def test_timeout_does_not_cancel_job(rig):
    service, provider, batch = rig
    service.submit_batch(batch["batch_id"])
    with pytest.raises(TimeoutError):
        service.wait(batch["run_ids"][0], timeout=0)
    assert provider.jobs[batch["run_ids"][0]]["observation"].state == "queued"


def test_partial_batch_resume_and_preflight_failure(rig):
    service, provider, _ = rig
    batch = service.prepare([EXAMPLE, EXAMPLE], provider, "a" * 64)
    service.submit(batch["run_ids"][0])
    provider.fail_preflight = True
    response = service.submit_batch(batch["batch_id"])
    assert [r["state"] for r in response["runs"]] == ["queued", "failed"]
    service.submit_batch(batch["batch_id"])
    assert provider.submissions == 1
