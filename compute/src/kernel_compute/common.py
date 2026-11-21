"""Dependency-free helpers also shipped to the GPU worker."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = {"succeeded", "failed", "timed_out", "cancelled"}


def now():
    return datetime.now(UTC).isoformat()


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def failure_result(request, status, reason, message, stage="provider"):
    return {
        "schema_version": 1,
        "run_id": request["run_id"],
        "request_hash": request["request_hash"],
        "task_hash": request["task_hash"],
        "worker_hash": request["worker_hash"],
        "environment_lock_hash": request["environment_lock_hash"],
        "task_name": request["task"]["name"],
        "worker_version": "0.1.0",
        "protocol_version": "cuda-event-warm-v1",
        "execution": {
            "status": status,
            "error": {"stage": stage, "reason": reason, "message": message},
        },
        "validation": {"status": "not_run"},
        "latency": None,
        "memory": None,
        "unavailable_reasons": {"latency": reason, "memory": reason},
        "environment": {},
        "warnings": [],
    }
