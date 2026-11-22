"""Provider-independent execution lifecycle, suitable for use by a future server."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from kernel_compute import bundle
from kernel_compute.common import TERMINAL, atomic_json, failure_result
from kernel_compute.models import validate_result
from kernel_compute.providers.base import ProviderError, RejectedSubmission
from kernel_compute.store import RunStore


class ExecutionService:
    def __init__(self, store: RunStore, provider_factory):
        self.store = store
        self.provider_factory = provider_factory

    def prepare(self, manifests: list[Path], provider, lock_hash: str, attempt_of=None):
        records = []
        # Snapshot the whole batch before any remote side effects.
        for manifest in manifests:
            run_id = uuid.uuid4().hex
            request = bundle.build(
                manifest,
                run_id,
                provider.id,
                provider.snapshot(),
                lock_hash,
                self.store.directory(run_id) / "bundle.tar",
                attempt_of,
            )
            record = {
                "run_id": run_id,
                "state": "prepared",
                "request": request,
                "handle": None,
                "observation": None,
                "error": None,
                "result_available": False,
            }
            self.store.save(record)
            records.append(record)
        batch = {"batch_id": uuid.uuid4().hex, "run_ids": [r["run_id"] for r in records]}
        self.store.save_batch(batch)
        return batch

    def provider(self, record):
        request = record["request"]
        return self.provider_factory(request["provider"], request["provider_config"])

    def submit(self, run_id):
        with self.store.lock(run_id):
            record = self.store.get(run_id)
            if record["state"] != "prepared":
                return record  # Recovery is observe(), never a second submit().
            provider = self.provider(record)
            try:
                provider.preflight(record["request"])
            except (ProviderError, ValueError) as exc:
                self.finish(
                    record,
                    failure_result(record["request"], "failed", "preflight_failed", str(exc)),
                )
                return record
            record["state"] = "submitting"
            self.store.save(record)
            try:
                handle = provider.submit(
                    record["request"], self.store.directory(run_id) / "bundle.tar"
                )
            except RejectedSubmission as exc:
                self.finish(
                    record,
                    failure_result(record["request"], "failed", "submission_rejected", str(exc)),
                )
                return record
            except ProviderError as exc:
                record["error"] = str(exc)
                record["observation"] = {
                    "state": "unknown",
                    "reason": "submission acknowledgement unavailable",
                }
                self.store.save(record)
                return record
            record["handle"] = handle.model_dump()
            record["state"] = "queued"
            self.store.save(record)
            return record

    def submit_batch(self, batch_id):
        batch = self.store.get_batch(batch_id)
        records = []
        for run_id in batch["run_ids"]:
            record = self.submit(run_id)
            if record["state"] == "submitting":
                record = self.status(run_id)
            records.append(record)
        return {**batch, "runs": [self.summary(r) for r in records]}

    @staticmethod
    def summary(record):
        return {
            key: record.get(key)
            for key in ("run_id", "state", "handle", "observation", "error", "result_available")
        }

    def finish(self, record, result):
        validate_result(result, record["request"])
        result["execution"]["provider"] = record["request"]["provider"]
        result["execution"]["handle"] = record["handle"]
        result["execution"]["scheduler"] = record["observation"]
        result["execution"]["requested_resources"] = record["request"]["task"]["resources"]
        times = (record.get("observation") or {}).get("timestamps", {})
        for field, first, last in (
            ("queue_wall_ms", "submitted", "started"),
            ("allocation_wall_ms", "started", "ended"),
        ):
            result["execution"][field] = None
            try:
                if times.get("timezone") == "UTC":
                    duration = (
                        datetime.fromisoformat(times[last]) - datetime.fromisoformat(times[first])
                    ).total_seconds() * 1000
                    result["execution"][field] = duration if duration >= 0 else None
            except (KeyError, ValueError):
                pass  # Slurm may report Unknown/None when a job never started.
        atomic_json(self.store.directory(record["run_id"]) / "result.json", result)
        record["state"] = result["execution"]["status"]
        record["error"] = result["execution"].get("error")
        record["result_available"] = True
        self.store.save(record)

    def status(self, run_id):
        with self.store.lock(run_id):
            record = self.store.get(run_id)
            if record["state"] in TERMINAL or record["state"] == "prepared":
                return record
            provider = self.provider(record)
            try:
                handle, observation = provider.observe(record["request"])
                record["observation"] = observation.model_dump()
                if handle:
                    record["handle"] = handle.model_dump()
                record["error"] = None
                if observation.state in {"queued", "running"}:
                    record["state"] = observation.state
                elif observation.state in TERMINAL:
                    try:
                        result = provider.fetch(
                            record["request"], self.store.directory(run_id) / "artifacts"
                        )
                        if result is not None:
                            validate_result(result, record["request"])
                    except ValueError as exc:
                        result = failure_result(
                            record["request"], "failed", "invalid_result", str(exc)
                        )
                    if observation.state in {"cancelled", "timed_out"}:
                        result = failure_result(
                            record["request"],
                            observation.state,
                            "scheduler_" + observation.state,
                            observation.raw_state or observation.state,
                        )
                    elif result is None:
                        # Shared storage/accounting visibility may briefly disagree.
                        since = record.setdefault("missing_result_since", time.time())
                        if time.time() - since < 30:
                            record["error"] = (
                                "scheduler finished; waiting for finalized worker artifacts"
                            )
                            self.store.save(record)
                            return record
                        reason = (
                            "out_of_memory"
                            if observation.raw_state == "OUT_OF_MEMORY"
                            else "missing_result"
                        )
                        result = failure_result(
                            record["request"],
                            "failed",
                            reason,
                            f"{observation.raw_state}: no valid finalized worker result",
                        )
                    elif (
                        observation.state == "failed"
                        and result["execution"]["status"] == "succeeded"
                    ):
                        result = failure_result(
                            record["request"],
                            "failed",
                            "scheduler_failed",
                            observation.raw_state or "scheduler failure",
                        )
                    self.finish(record, result)
                    return record
            except ProviderError as exc:
                record["observation"] = {"state": "unknown", "reason": str(exc)}
                record["error"] = str(exc)
            self.store.save(record)
            return record

    def result(self, run_id):
        record = self.status(run_id)
        if not record["result_available"]:
            raise ValueError(f"result not ready ({record['state']}); run ID: {run_id}")
        result = json.loads((self.store.directory(run_id) / "result.json").read_text())
        return validate_result(result, record["request"])

    def wait(self, run_id, timeout=None, interval=15):
        deadline = time.monotonic() + timeout if timeout is not None else None
        failures = 0
        while True:
            record = self.status(run_id)
            if record["state"] in TERMINAL:
                return record
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"wait timed out; job continues; run ID: {run_id}")
            failures = (
                min(failures + 1, 3)
                if (record.get("observation") or {}).get("state") == "unknown"
                else 0
            )
            delay = min(60, interval * 2**failures)
            if deadline is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            time.sleep(delay)

    def cancel(self, run_id):
        with self.store.lock(run_id):
            record = self.store.get(run_id)
            if record["state"] in TERMINAL:
                return {"requested": False, "state": record["state"]}
            if record["state"] == "prepared":
                self.finish(
                    record,
                    failure_result(
                        record["request"],
                        "cancelled",
                        "cancelled_before_submission",
                        "cancelled before remote submission",
                    ),
                )
                return {"requested": True, "state": "cancelled"}
            return self.provider(record).cancel(record["request"])

    def logs(self, run_id, cursor=None):
        record = self.store.get(run_id)
        return self.provider(record).read_logs(record["request"], cursor)
