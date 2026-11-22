"""Provider adapters; worker code does not import these modules."""

from pathlib import Path

WORKER_LOCK = Path(__file__).with_name("worker.lock")


def environment_lock():
    return WORKER_LOCK.read_bytes()
