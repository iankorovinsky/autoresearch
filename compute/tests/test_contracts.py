import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from kernel_compute.bundle import build, snapshot_task
from kernel_compute.models import Task, WaterlooConfig, safe_relative
from kernel_compute.providers.remote import unpack
from kernel_compute.worker.measure import summarize

EXAMPLE = Path(__file__).parents[1] / "examples/model-new/task.toml"


@pytest.mark.parametrize(
    "name",
    ["../secret", "/tmp/x", ".env", "foo/.ssh/id_rsa", "a.pem", "a/../b", "a\\b", "node_modules/x"],
)
def test_bundle_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_snapshot_is_frozen_and_contains_no_repo(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(EXAMPLE.read_text())
    (task_dir / "task.py").write_text("original = True")
    (task_dir / ".env").write_text("do not upload")
    destination = tmp_path / "bundle.tar"
    request = build(task_dir / "task.toml", "a" * 32, "fake", {}, "b" * 64, destination)
    (task_dir / "task.py").write_text("changed = True")
    with tarfile.open(destination) as archive:
        assert archive.extractfile("task/task.py").read() == b"original = True"
        assert not any(".env" in name for name in archive.getnames())
        assert (
            json.load(archive.extractfile("request.json"))["request_hash"]
            == request["request_hash"]
        )


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "task.toml").write_text(EXAMPLE.read_text())
    (tmp_path / "task.py").symlink_to(EXAMPLE)
    with pytest.raises(ValueError, match="regular file"):
        snapshot_task(tmp_path / "task.toml")


@pytest.mark.parametrize("kind", ["traversal", "link", "duplicate"])
def test_remote_archive_validation(tmp_path, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        member = tarfile.TarInfo("../escape" if kind == "traversal" else "file")
        if kind == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        archive.addfile(member)
        if kind == "duplicate":
            archive.addfile(member)
    with pytest.raises(ValueError):
        unpack(data.getvalue(), tmp_path)


def test_explicit_validation_and_supported_protocol():
    base = {"name": "x", "entrypoint": "x.py:build", "files": ["x.py"]}
    with pytest.raises(ValueError, match="explicit"):
        Task(**base, validation={"mode": "reference"})
    with pytest.raises(ValueError):
        Task(**base, measurement={"cache_policy": "cold"})
    with pytest.raises(ValueError):
        Task(**base, resources={"gpu_count": 2})
    with pytest.raises(ValueError):
        WaterlooConfig(user="-oProxyCommand=bad")


def test_known_statistics_and_units():
    summary = summarize(list(range(1, 101)))
    assert summary["median"] == 50.5
    assert summary["p95"] == 95
    assert summary["units"] == "us"
    assert summary["stddev"] == pytest.approx(math.sqrt(833.25))
    for bad in ([1, float("nan")], [-1, 1], [1]):
        with pytest.raises(ValueError):
            summarize(bad)


def test_cli_does_not_import_gpu_packages():
    output = subprocess.check_output(
        [sys.executable, "-c", "import kernel_compute.cli,sys; print('torch' in sys.modules)"],
        text=True,
    )
    assert output.strip() == "False"


def test_invalid_poll_interval_does_not_submit(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kernel_compute.cli",
            "run",
            str(EXAMPLE),
            "--poll-interval",
            "0",
            "--state-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "poll interval" in json.loads(proc.stdout)["error"]
    assert not (tmp_path / "runs.sqlite3").exists()
