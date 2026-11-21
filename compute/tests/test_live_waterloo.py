"""Opt-in, bounded real cluster acceptance tests. Never run in the ordinary suite.

KCOMPUTE_LIVE=1 KCOMPUTE_SSH_USER=... uv run pytest -m live -s
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kernel_compute.common import TERMINAL, atomic_json, digest
from kernel_compute.models import WaterlooConfig
from kernel_compute.providers.base import ProviderError
from kernel_compute.providers.waterloo_datasci import WaterlooProvider, environment_lock
from kernel_compute.service import ExecutionService
from kernel_compute.store import RunStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("KCOMPUTE_LIVE") != "1", reason="live cluster tests require opt-in"
    ),
]
ROOT = Path(__file__).parents[1]
REPORT = []


@pytest.fixture
def live():
    config = WaterlooConfig(user=os.environ.get("KCOMPUTE_SSH_USER"))
    assert config.node == "gpu-8-vm"
    provider = WaterlooProvider(config)
    store = RunStore()
    service = ExecutionService(store, lambda _, cfg: WaterlooProvider(WaterlooConfig(**cfg)))
    tracked = []
    yield service, provider, tracked
    for run_id in tracked:
        record = service.status(run_id)
        if record["state"] not in TERMINAL:
            service.cancel(run_id)
            record = service.wait(run_id, timeout=90, interval=5)
        result = service.result(run_id)
        REPORT.append(
            {
                "run_id": run_id,
                "task": record["request"]["task"]["name"],
                "task_hash": record["request"]["task_hash"],
                "worker_hash": record["request"]["worker_hash"],
                "native_id": (record["handle"] or {}).get("native_id"),
                "result": result,
            }
        )
    atomic_json(
        Path(os.environ.get("KCOMPUTE_LIVE_REPORT", "/tmp/kcompute-live-results.json")), REPORT
    )


def submit(live, manifests):
    service, provider, tracked = live
    batch = service.prepare(manifests, provider, digest(environment_lock()))
    tracked.extend(batch["run_ids"])
    service.submit_batch(batch["batch_id"])
    return batch


def fixture_task(tmp_path, example="model-new", name="test"):
    destination = tmp_path / name
    shutil.copytree(
        ROOT / "examples" / example, destination, ignore=shutil.ignore_patterns("__pycache__")
    )
    manifest = destination / "task.toml"
    manifest.write_text(
        manifest.read_text()
        .replace('name = "model-new-relu"', f'name = "{name}"')
        .replace('name = "cuda-vector-add-fp32"', f'name = "{name}"')
    )
    return manifest


def wait_result(service, run_id):
    service.wait(run_id, timeout=240, interval=5)
    return service.result(run_id)


def test_cuda_and_fresh_process_retrieval(live):
    service, _, tracked = live
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kernel_compute.cli",
            "submit",
            str(ROOT / "examples/vector-add/task.toml"),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    submission = json.loads(proc.stdout)
    run_id = submission["run_id"]
    tracked.append(run_id)
    result = wait_result(service, run_id)
    assert result["execution"]["status"] == "succeeded"
    assert result["validation"]["passed_seeds"] == [42, 43, 44]
    assert result["environment"]["node"] == "gpu-8-vm"
    assert result["environment"]["mig"] is False
    assert result["latency"]["median"] > 0 and result["latency"]["sample_count"] == 100
    assert result["memory"]["peak_allocated"] >= 3 * 1048576 * 4
    fresh = subprocess.check_output(
        [sys.executable, "-m", "kernel_compute.cli", "result", run_id, "--json"], text=True
    )
    assert json.loads(fresh)["request_hash"] == result["request_hash"]
    samples = json.loads((service.store.directory(run_id) / "artifacts/samples.json").read_text())
    assert len(samples["samples"]) == 100


def test_batch_wrong_output_and_compile_failure(live, tmp_path):
    wrong = fixture_task(tmp_path, name="live-wrong-output")
    source = wrong.with_name("task.py")
    source.write_text(source.read_text().replace("return torch.clamp_min(x, 0)", "return x + 100"))
    bad = fixture_task(tmp_path, "vector-add", "live-compile-error")
    with bad.with_name("kernel.cu").open("a") as stream:
        stream.write("\nthis is deliberately invalid CUDA source;\n")
    batch = submit(live, [wrong, bad])
    service = live[0]
    results = [wait_result(service, run_id) for run_id in batch["run_ids"]]
    assert [r["execution"]["error"]["reason"] for r in results] == [
        "validation_failed",
        "compile_error",
    ]
    assert all(r["latency"] is None for r in results)
    assert results[0]["validation"]["status"] == "failed"
    assert results[1]["validation"]["status"] == "not_run"


def test_worker_timeout(live, tmp_path):
    manifest = fixture_task(tmp_path, name="live-timeout")
    manifest.write_text(manifest.read_text().replace("wall_time_s = 90", "wall_time_s = 15"))
    source = manifest.with_name("task.py")
    source.write_text(
        source.read_text().replace(
            "return Task(case, device)",
            "import time\n    time.sleep(120)\n    return Task(case, device)",
        )
    )
    run_id = submit(live, [manifest])["run_ids"][0]
    result = wait_result(live[0], run_id)
    assert result["execution"]["status"] == "timed_out"
    assert result["execution"]["error"]["reason"] in {"worker_timeout", "scheduler_timed_out"}
    assert result["latency"] is None


def test_resource_requirement_rejected_on_worker(live, tmp_path):
    manifest = fixture_task(tmp_path, name="live-insufficient-vram")
    manifest.write_text(
        manifest.read_text().replace("[resources]", "[resources]\nmin_gpu_memory_gib = 1000")
    )
    run_id = submit(live, [manifest])["run_ids"][0]
    result = wait_result(live[0], run_id)
    assert result["execution"]["error"]["reason"] == "environment_mismatch"
    assert "insufficient memory" in result["execution"]["error"]["message"]


def test_pending_and_running_cancellation(live, tmp_path):
    service = live[0]
    blocker = fixture_task(tmp_path, name="live-cancel-running")
    source = blocker.with_name("task.py")
    source.write_text(
        source.read_text().replace(
            "return Task(case, device)",
            "import time\n    time.sleep(120)\n    return Task(case, device)",
        )
    )
    blocker_id = submit(live, [blocker])["run_ids"][0]
    deadline = time.monotonic() + 90
    while service.status(blocker_id)["state"] != "running":
        assert time.monotonic() < deadline
        time.sleep(2)
    queued_id = submit(live, [ROOT / "examples/model-new/task.toml"])["run_ids"][0]
    assert service.status(queued_id)["state"] == "queued"
    service.cancel(queued_id)
    assert wait_result(service, queued_id)["execution"]["status"] == "cancelled"
    service.cancel(blocker_id)
    assert wait_result(service, blocker_id)["execution"]["status"] == "cancelled"


def test_lost_ack_recovers_real_job_once(live):
    service, provider, tracked = live
    original = provider.rpc
    lost = False

    def lose_response(operation, payload, data=None):
        nonlocal lost
        response = original(operation, payload, data)
        if operation == "submit" and not lost:
            lost = True
            raise ProviderError("injected loss after actual sbatch acceptance")
        return response

    provider.rpc = lose_response
    interrupted = ExecutionService(service.store, lambda *_: provider)
    batch = interrupted.prepare(
        [ROOT / "examples/model-new/task.toml"], provider, digest(environment_lock())
    )
    tracked.extend(batch["run_ids"])
    run_id = batch["run_ids"][0]
    assert interrupted.submit(run_id)["state"] == "submitting"
    recovered = service.status(run_id)
    native_id = recovered["handle"]["native_id"]
    service.submit_batch(batch["batch_id"])
    result = wait_result(service, run_id)
    assert result["execution"]["status"] == "succeeded"
    assert result["execution"]["handle"]["native_id"] == native_id
    assert lost


def test_unvalidated_profiling_is_labeled(live, tmp_path):
    manifest = fixture_task(tmp_path, name="live-unvalidated")
    manifest.write_text(manifest.read_text().split("[validation]")[0])
    run_id = submit(live, [manifest])["run_ids"][0]
    result = wait_result(live[0], run_id)
    assert result["execution"]["status"] == "succeeded"
    assert result["validation"]["status"] == "not_requested"
    assert "speedup" not in result
