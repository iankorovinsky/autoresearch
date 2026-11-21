import io
import json
import tarfile
import tomllib
from pathlib import Path

from kernel_compute.common import digest, json_bytes, now
from kernel_compute.models import Task

MAX_BUNDLE_BYTES = 32 * 1024**2


def snapshot_task(manifest: Path):
    task = Task.model_validate(tomllib.loads(manifest.read_text()))
    root = manifest.parent.resolve()
    files = {}
    for name in task.files:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"bundle file must be a regular file within task directory: {name}")
        if path.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("bundle file too large")
        files[f"task/{name}"] = path.read_bytes()
    if sum(map(len, files.values())) > MAX_BUNDLE_BYTES:
        raise ValueError("task exceeds 32 MiB bundle limit")
    return task, files


def build(
    manifest: Path,
    run_id: str,
    provider: str,
    provider_config: dict,
    lock_hash: str,
    destination: Path,
    attempt_of: str | None = None,
):
    task, files = snapshot_task(manifest)
    source = Path(__file__).parent
    worker_files = [
        source / "__init__.py",
        source / "common.py",
        *sorted((source / "worker").glob("*.py")),
    ]
    worker_hashes = {}
    for path in worker_files:
        name = "worker/kernel_compute/" + str(path.relative_to(source))
        files[name] = path.read_bytes()
        worker_hashes[name] = digest(files[name])
    task_hash = digest(
        json_bytes(
            {
                "task": task.model_dump(),
                "files": {
                    name: digest(data) for name, data in files.items() if name.startswith("task/")
                },
            }
        )
    )
    request = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": now(),
        "task": task.model_dump(),
        "task_hash": task_hash,
        "worker_hash": digest(json_bytes(worker_hashes)),
        "environment_lock_hash": lock_hash,
        "provider": provider,
        "provider_config": provider_config,
        "attempt_of": attempt_of,
    }
    request["request_hash"] = digest(json_bytes(request))
    files["request.json"] = json_bytes(request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    if destination.stat().st_size > MAX_BUNDLE_BYTES:
        destination.unlink()
        raise ValueError("complete bundle exceeds 32 MiB")
    return request


def read_request(bundle: Path):
    with tarfile.open(bundle) as archive:
        return json.load(archive.extractfile("request.json"))
