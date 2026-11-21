import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from kernel_compute.common import digest
from kernel_compute.models import WaterlooConfig
from kernel_compute.providers import remote
from kernel_compute.providers.base import InvalidArtifact
from kernel_compute.providers.waterloo_datasci import WaterlooProvider


def test_scheduler_owner_name_and_parent_filter(monkeypatch):
    calls = []

    def run(args):
        calls.append(args)
        if args[0] == "squeue":
            return "8116|RUNNING|someone-else|gpu-8-vm"
        return "8116|COMPLETED|0:0|kc-mine|2026-11-21T21:49:04|2026-11-21T21:49:04|2026-11-21T21:49:09\n8116.batch|COMPLETED|0:0|batch|||"

    monkeypatch.setattr(remote, "run", run)
    found = remote.records("kc-mine", "2026-11-21T21:49:02+00:00", "8116")
    assert len(found) == 1 and found[0]["exit_code"] == "0:0"
    assert "--allocations" in calls[1] and "--user" in calls[1]


def test_remote_intent_never_resubmits(tmp_path, monkeypatch):
    config = {"remote_root": ".cache/kcompute", "host": "test"}
    directory = tmp_path / ".cache/kcompute/runs" / ("a" * 32)
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(
        json.dumps(
            {
                "run_id": "a" * 32,
                "request_hash": "b" * 64,
                "created_at": "2026-11-21T00:00:00+00:00",
            }
        )
    )
    (directory / "intent.json").write_text("{}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(remote, "records", lambda *args: [])
    monkeypatch.setattr(remote, "run", lambda args: pytest.fail("must not call sbatch"))
    result = remote.submit(config, {"run_id": "a" * 32, "request_hash": "b" * 64})
    assert result["ambiguous"]


def test_fetch_checksum_and_resume(tmp_path, monkeypatch):
    provider = WaterlooProvider(WaterlooConfig())
    request = {"run_id": "a" * 32, "request_hash": "b" * 64}
    contents = {"request.json": b"{}", "result.json": b'{"valid": true}'}
    manifest = {
        **request,
        "files": {
            name: {"size": len(data), "sha256": digest(data)} for name, data in contents.items()
        },
    }
    calls = []

    def rpc(op, payload):
        calls.append(payload)
        return {
            "complete": True,
            "manifest": manifest,
            "files": {
                name: base64.b64encode(data).decode()
                for name, data in contents.items()
                if name not in payload["have"]
            },
        }

    monkeypatch.setattr(provider, "rpc", rpc)
    assert provider.fetch(request, tmp_path) == {"valid": True}
    provider.fetch(request, tmp_path)
    assert set(calls[-1]["have"]) == set(contents)
    contents["result.json"] = b"corrupted"
    (tmp_path / "result.json").unlink()
    with pytest.raises(InvalidArtifact, match="checksum"):
        provider.fetch(request, tmp_path)


def test_stage_rejects_reusing_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = {"remote_root": ".cache/kcompute"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        info = tarfile.TarInfo("request.json")
        archive.addfile(info, io.BytesIO())
    data = buf.getvalue()
    payload = {"run_id": "a" * 32, "bundle_hash": digest(data)}
    assert remote.stage(config, payload, data)["staged"]
    changed = data + b"different"
    with pytest.raises(ValueError, match="different bundle"):
        remote.stage(config, {**payload, "bundle_hash": digest(changed)}, changed)
