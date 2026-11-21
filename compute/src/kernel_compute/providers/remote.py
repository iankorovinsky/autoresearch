"""Dependency-free SSH RPC helper. Executed under the authenticated user's identity.

Only this adapter knows Slurm. A persistent intent prevents unsafe automatic
resubmission after lost acknowledgements; receipts survive the coordinator.
"""

import base64
import fcntl
import getpass
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def run(args):
    proc = subprocess.run(
        args,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={**os.environ, "TZ": "UTC"},
    )
    if proc.returncode:
        raise RuntimeError(f"{args[0]} failed ({proc.returncode}): {proc.stderr.strip()[:4096]}")
    return proc.stdout.strip()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def root_path(config):
    root = Path.home() / config["remote_root"]
    if (
        not root.resolve().is_relative_to(Path.home().resolve())
        or root.resolve() == Path.home().resolve()
    ):
        raise ValueError("workspace must be under remote HOME")
    return root


def run_path(config, run_id):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("invalid run ID")
    return root_path(config) / "runs" / run_id


def environment_path(config, lock_hash):
    if not re.fullmatch(r"[a-f0-9]{64}", lock_hash):
        raise ValueError("invalid environment hash")
    return root_path(config) / "envs" / lock_hash


def unpack(data, destination):
    if len(data) > 32 * 1024**2:
        raise ValueError("bundle too large")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) > 600 or sum(m.size for m in members) > 32 * 1024**2:
            raise ValueError("bundle exceeds limits")
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in seen
            ):
                raise ValueError("unsafe or duplicate archive member")
            seen.add(member.name)
        for member in members:
            path = destination / member.name
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("archive path escapes run directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())


def records(job_name, submitted_at, job_id=None):
    """Owner-scoped records; full job names prevent job-ID reuse confusion."""
    user = getpass.getuser()
    queued = run(["squeue", "--user", user, "--noheader", "--format=%i|%T|%j|%R"])
    found = []
    for line in queued.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4 and parts[2] == job_name and (job_id is None or parts[0] == job_id):
            found.append(
                {"id": parts[0], "raw_state": parts[1], "reason": parts[3], "timestamps": {}}
            )
    if found:
        return found
    # Start from the recorded submission date, not sacct's changing default window.
    args = [
        "sacct",
        "--user",
        user,
        "--allocations",
        "--noheader",
        "--parsable2",
        "--starttime",
        submitted_at[:10],
        "--format=JobIDRaw,State,ExitCode,JobName%128,Submit,Start,End",
    ]
    if job_id:
        args += ["--jobs", job_id]
    for line in run(args).splitlines():
        parts = line.split("|")
        if len(parts) >= 7 and parts[3] == job_name and (job_id is None or parts[0] == job_id):
            found.append(
                {
                    "id": parts[0],
                    "raw_state": parts[1],
                    "exit_code": parts[2],
                    "reason": None,
                    "timestamps": dict(zip(("submitted", "started", "ended"), parts[4:7])),
                }
            )
    return found


def handle(config, request, job_id):
    return {
        "provider": "waterloo-datasci",
        "run_id": request["run_id"],
        "native_id": job_id,
        "cluster": config["host"],
        "submitted_at": request["created_at"],
        "artifact_locator": str(run_path(config, request["run_id"])),
    }


def recover(config, directory):
    receipt = directory / "receipt.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    if not (directory / "intent.json").exists():
        return None
    request = json.loads((directory / "request.json").read_text())
    matches = records("kc-" + request["run_id"], request["created_at"])
    if len(matches) == 1:
        value = handle(config, request, matches[0]["id"])
        write_json(receipt, value)
        return value
    return None


def stage(config, payload, data):
    directory = run_path(config, payload["run_id"])
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(directory / ".lock"):
        manifest_path = directory / "bundle.sha256"
        expected = payload["bundle_hash"]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("bundle transport checksum mismatch")
        if manifest_path.exists():
            if manifest_path.read_text() != expected:
                raise ValueError("run ID already has a different bundle")
            return {"staged": True}
        if (directory / "intent.json").exists():
            raise ValueError("cannot replace an attempted submission")
        unpack(data, directory)
        manifest_path.write_text(expected)
        return {"staged": True}


def submit(config, payload):
    directory = run_path(config, payload["run_id"])
    with locked(directory / ".lock"):
        request = json.loads((directory / "request.json").read_text())
        if request["request_hash"] != payload["request_hash"]:
            raise ValueError("request hash mismatch")
        previous = recover(config, directory)
        if previous:
            return {"handle": previous}
        if (directory / "intent.json").exists():
            return {
                "ambiguous": True,
                "message": "submission intent exists without a unique receipt; not resubmitting",
            }
        env = environment_path(config, request["environment_lock_hash"])
        if not (env / "ready.json").exists():
            raise ValueError("environment is not ready; run kcompute setup")
        resources = request["task"]["resources"]
        quote = shlex.quote
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -eu",
                "umask 077",
                "ulimit -c 0",
                "export CUDA_HOME=" + quote(config["cuda_home"]),
                'export PATH="$CUDA_HOME/bin:' + str(env / "bin") + ':$PATH"',
                "export PYTHONPATH=" + quote(str(directory / "worker")),
                "export PYTHONUNBUFFERED=1",
                "export OMP_NUM_THREADS=" + str(resources["cpu_cores"]),
                "export TORCH_EXTENSIONS_DIR=" + quote(str(directory / "build")),
                "export TRITON_CACHE_DIR=" + quote(str(directory / "triton")),
                "export CUDA_CACHE_PATH=" + quote(str(directory / "cuda-cache")),
                "exec " + shlex.join([str(env / "bin/python"), "-m", "kernel_compute.worker.main"]),
                "",
            ]
        )
        batch = directory / "job.sh"
        batch.write_text(script)
        write_json(
            directory / "intent.json",
            {"request_hash": request["request_hash"], "at": datetime.now(UTC).isoformat()},
        )
        seconds = resources["wall_time_s"]
        wall = f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"
        args = [
            "sbatch",
            "--parsable",
            "--partition",
            config["partition"],
            "--nodelist",
            config["node"],
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:1",
            "--cpus-per-task",
            str(resources["cpu_cores"]),
            "--mem",
            f"{resources['host_memory_gib']}G",
            "--time",
            wall,
            "--job-name",
            "kc-" + request["run_id"],
            "--chdir",
            str(directory),
            "--output",
            str(directory / "slurm.out"),
            "--error",
            str(directory / "slurm.err"),
            str(batch),
        ]
        response = run(args)
        job_id = response.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError("unrecognized sbatch response; submission remains ambiguous")
        receipt = handle(config, request, job_id)
        write_json(directory / "receipt.json", receipt)
        return {"handle": receipt}


def observe(config, payload):
    directory = run_path(config, payload["run_id"])
    with locked(directory / ".lock"):
        receipt = recover(config, directory)
    if not receipt:
        return {
            "handle": None,
            "observation": {"state": "unknown", "reason": "submission not yet recoverable"},
        }
    found = records("kc-" + receipt["run_id"], receipt["submitted_at"], receipt["native_id"])
    if len(found) != 1:
        return {
            "handle": receipt,
            "observation": {
                "state": "unknown",
                "reason": "scheduler/accounting has no unique matching record",
            },
        }
    record = found[0]
    raw = record["raw_state"].split()[0].rstrip("+")
    states = {
        "PENDING": "queued",
        "CONFIGURING": "queued",
        "RUNNING": "running",
        "COMPLETING": "running",
        "SUSPENDED": "running",
        "COMPLETED": "succeeded",
        "CANCELLED": "cancelled",
        "TIMEOUT": "timed_out",
        "FAILED": "failed",
        "OUT_OF_MEMORY": "failed",
        "NODE_FAIL": "failed",
        "PREEMPTED": "failed",
        "BOOT_FAIL": "failed",
        "DEADLINE": "timed_out",
        "REVOKED": "failed",
    }
    record.pop("id")
    record["state"] = states.get(raw, "unknown")
    record["timestamps"]["timezone"] = "UTC"
    return {"handle": receipt, "observation": record}


def logs(config, payload):
    directory = run_path(config, payload["run_id"])
    cursors = payload.get("cursor") or {}
    result = {"chunks": {}, "cursor": {}}
    for name in ("stdout.log", "stderr.log", "slurm.out", "slurm.err"):
        path = directory / name
        offset = max(0, int(cursors.get(name, 0)))
        if path.exists():
            with path.open("rb") as stream:
                if offset > path.stat().st_size:
                    offset = 0
                stream.seek(offset)
                data = stream.read(65536)
                result["chunks"][name] = data.decode("utf-8", errors="replace")
                offset = stream.tell()
        result["cursor"][name] = offset
    return result


def fetch(config, payload):
    directory = run_path(config, payload["run_id"])
    completion = directory / "completion.json"
    if not completion.exists():
        return {"complete": False}
    manifest = json.loads(completion.read_text())
    allowed = {"request.json", "result.json", "samples.json", "stdout.log", "stderr.log"}
    files = {}
    for name, entry in manifest["files"].items():
        if name not in allowed or entry["size"] > 4 * 1024**2:
            raise ValueError("invalid artifact manifest")
        data = (directory / name).read_bytes()
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("artifact checksum mismatch")
        if payload.get("have", {}).get(name) != entry["sha256"]:
            files[name] = base64.b64encode(data).decode()
    return {"complete": True, "manifest": manifest, "files": files}


def setup(config, payload):
    env = environment_path(config, payload["lock_hash"])
    env.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(env / ".submit.lock"):
        if (env / "ready.json").exists():
            return {"ready": True, "environment": json.loads((env / "ready.json").read_text())}
        if (env / "setup-receipt.json").exists():
            return json.loads((env / "setup-receipt.json").read_text())
        if (env / "setup-intent.json").exists():
            raise RuntimeError("setup submission ambiguous; inspect setup records before retrying")
        lock_data = base64.b64decode(payload["lock"])
        if hashlib.sha256(lock_data).hexdigest() != payload["lock_hash"]:
            raise ValueError("environment lock hash mismatch")
        (env / "worker.lock").write_bytes(lock_data)
        (env / "bootstrap.py").write_text(payload["bootstrap"])
        script = env / "setup.sh"
        script.write_text(
            "#!/bin/bash\nset -eu\numask 077\nexport CUDA_HOME="
            + shlex.quote(config["cuda_home"])
            + "\nexec "
            + shlex.join(["python3", str(env / "bootstrap.py"), str(env)])
            + "\n"
        )
        submitted_at = datetime.now(UTC).isoformat()
        write_json(env / "setup-intent.json", {"at": submitted_at})
        native = run(
            [
                "sbatch",
                "--parsable",
                "--partition",
                config["partition"],
                "--nodelist",
                config["node"],
                "--gres=gpu:1",
                "--cpus-per-task=4",
                "--mem=8G",
                "--time=00:20:00",
                "--job-name",
                "kc-setup-" + payload["lock_hash"][:16],
                "--output",
                str(env / "setup.out"),
                "--error",
                str(env / "setup.err"),
                str(script),
            ]
        ).split(";")[0]
        value = {
            "ready": False,
            "native_id": native,
            "submitted_at": submitted_at,
            "path": str(env),
        }
        write_json(env / "setup-receipt.json", value)
        return value


def dispatch(config, operation, payload):
    if operation == "stage":
        return stage(config, payload, sys.stdin.buffer.read(32 * 1024**2 + 1))
    if operation == "submit":
        return submit(config, payload)
    if operation == "observe":
        return observe(config, payload)
    if operation == "logs":
        return logs(config, payload)
    if operation == "fetch":
        return fetch(config, payload)
    if operation == "cancel":
        status = observe(config, payload)
        if status["observation"]["state"] == "unknown":
            raise RuntimeError("cannot verify job ownership/identity; cancellation not issued")
        if status["observation"]["state"] in {"queued", "running"}:
            run(["scancel", status["handle"]["native_id"]])
        return {"requested": True, **status}
    if operation == "doctor":
        env = environment_path(config, payload["lock_hash"])
        ready = (
            json.loads((env / "ready.json").read_text()) if (env / "ready.json").exists() else None
        )
        return {
            "host": config["host"],
            "user": getpass.getuser(),
            "slurm": run(["sinfo", "--version"]),
            "node": run(
                ["sinfo", "--noheader", "--Node", "--nodes", config["node"], "--format=%N|%T|%G"]
            ),
            "environment_ready": ready is not None,
            "environment": ready,
            "environment_path": str(env),
        }
    if operation == "setup":
        return setup(config, payload)
    if operation == "setup_status":
        env = environment_path(config, payload["lock_hash"])
        if (env / "ready.json").exists():
            return {"ready": True, "environment": json.loads((env / "ready.json").read_text())}
        receipt = json.loads((env / "setup-receipt.json").read_text())
        found = records(
            "kc-setup-" + payload["lock_hash"][:16], receipt["submitted_at"], receipt["native_id"]
        )
        return {
            "ready": False,
            **receipt,
            "records": found,
            "log_tail": {
                name: (env / name).read_text(errors="replace")[-8000:]
                if (env / name).exists()
                else ""
                for name in ("setup.out", "setup.err")
            },
        }
    raise ValueError("unsupported operation")


if __name__ == "__main__":
    os.umask(0o077)
    config, operation, payload = json.loads(sys.argv[1])
    try:
        print(json.dumps({"ok": True, "value": dispatch(config, operation, payload)}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "kind": type(exc).__name__}))
