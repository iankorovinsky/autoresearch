import fcntl
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from kernel_compute.common import now


def state_directory():
    return Path(os.environ.get("KCOMPUTE_STATE_DIR", Path.home() / ".local/state/kcompute"))


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = root or state_directory()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.root / "runs.sqlite3", timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def directory(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid run ID")
        return self.root / "runs" / run_id

    @contextmanager
    def lock(self, run_id):
        directory = self.directory(run_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def save(self, record):
        record["updated_at"] = now()
        with self.connection() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                (record["run_id"], json.dumps(record, allow_nan=False)),
            )

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT value FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown run: {run_id}")
        return json.loads(row[0])

    def list(self, provider=None):
        with self.connection() as db:
            records = [
                json.loads(row[0])
                for row in db.execute("SELECT value FROM runs ORDER BY rowid DESC")
            ]
        return [r for r in records if provider is None or r["request"]["provider"] == provider]

    def save_batch(self, batch):
        with self.connection() as db:
            db.execute("INSERT INTO batches VALUES (?, ?)", (batch["batch_id"], json.dumps(batch)))

    def get_batch(self, batch_id):
        with self.connection() as db:
            row = db.execute("SELECT value FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown batch: {batch_id}")
        return json.loads(row[0])
