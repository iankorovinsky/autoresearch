import json
import tarfile
from pathlib import Path

from kernel_compute.cli import main
from kernel_compute.common import digest, failure_result, json_bytes
from kernel_compute.providers import baseten_backend
from kernel_compute.providers.baseten_backend import BasetenProvider

PROJECT = {"name": "kcompute", "id": "project"}
JOBS = "/training_projects/project/jobs"
JOB_PATH = f"{JOBS}/job"


def test_provider_lists_published_and_workspace_gpu_options(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    instance = {"gpu_count": 1, "gpu_type": "WORKSPACE-GPU", "id": "workspace-gpu:4x16"}

    def api(self, path):
        assert path == "/instance_type_prices"
        return {
            "instance_types": [
                {"instance_type": instance, "price": 0.10},
                {"instance_type": {"gpu_count": 0, "gpu_type": None}, "price": 0.01},
            ]
        }

    monkeypatch.setattr(BasetenProvider, "api", api)
    assert (
        main(
            [
                "providers",
                "list",
                "--provider",
                "baseten",
                "--config",
                str(tmp_path / "missing.toml"),
            ]
        )
        == 0
    )
    (provider,) = json.loads(capsys.readouterr().out)
    assert {"T4", "L4", "A10G", "A100", "H100MIG", "H100", "H200", "B200", "RTX-PRO-6000"} <= set(
        provider["gpu_types"]
    )
    assert "WORKSPACE-GPU" in provider["gpu_types"]
    assert provider["workspace_gpu_instances"] == [{**instance, "usd_per_minute": 0.10}]


def test_t4_probe_uses_one_gpu_and_returns_checkpoint_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    report = {"actual_gpu": "Tesla T4", "tools": {"cuda-events": {"status": "passed"}}}
    requests = []

    def api(self, path, *, body=None):
        requests.append((path, body))
        if path == "/training_projects?name=kcompute":
            return {"training_projects": [PROJECT]}
        if path == "/blobs/credentials/train":
            return {}
        if path == JOBS:
            assert body["training_job"]["compute"] == {
                "node_count": 1,
                "cpu_count": 4,
                "memory": "16Gi",
                "accelerator": {"accelerator": "T4", "count": 1},
            }
            assert body["training_job"]["runtime"]["checkpointing_config"]["enabled"]
            return {"training_job": {**job, "current_status": "TRAINING_JOB_RUNNING"}}
        if path == JOB_PATH:
            return {"training_job": job}
        assert path == f"{JOB_PATH}/checkpoint_files"
        return {"presigned_urls": [{"relative_file_name": "probe.json", "node_rank": 0}]}

    job = {
        "id": "job",
        "training_project_id": "project",
        "current_status": "TRAINING_JOB_COMPLETED",
        "created_at": "2026-11-21T00:00:00Z",
    }

    def upload(archive, credentials):
        with tarfile.open(archive) as source:
            assert set(source.getnames()) == {"runner.py", "probe.py", "worker.lock"}
        return {"s3_bucket": "example", "s3_key": "probe"}

    monkeypatch.setattr(BasetenProvider, "api", api)
    monkeypatch.setattr(baseten_backend, "upload_archive", upload)
    monkeypatch.setattr(baseten_backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(BasetenProvider, "download", staticmethod(lambda _: json_bytes(report)))

    config = tmp_path / "config.toml"
    config.write_text('[providers.baseten]\ngpu = "L4"\n')
    assert main(["probe", "--provider", "baseten", "--gpu", "t4", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "provider": "baseten",
        "requested_gpu": "T4",
        "job_id": "job",
        **report,
    }
    assert [path for path, _ in requests] == [
        "/training_projects?name=kcompute",
        "/blobs/credentials/train",
        JOBS,
        JOB_PATH,
        f"{JOB_PATH}/checkpoint_files",
    ]


def test_baseten_cli_submission_and_checkpoint_retrieval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    job = {
        "id": "job",
        "training_project_id": "project",
        "current_status": "TRAINING_JOB_COMPLETED",
        "created_at": "2026-11-21T00:00:00Z",
    }
    data = {}

    def api(self, path, *, body=None):
        if path == "/training_projects?name=kcompute":
            return {"training_projects": [PROJECT]}
        if path == "/blobs/credentials/train":
            return {}
        if path == JOBS and body:
            job["name"] = body["training_job"]["name"]
            assert body["training_job"]["compute"]["accelerator"]["accelerator"] == "L4"
            return {"training_job": job}
        if path == JOBS:
            return {"training_jobs": [job]}
        assert path == f"{JOB_PATH}/checkpoint_files"
        return {
            "presigned_urls": [{"relative_file_name": key, "node_rank": 0} for key in data],
        }

    def upload(archive, credentials):
        with tarfile.open(archive) as source:
            with tarfile.open(fileobj=source.extractfile("bundle.tar")) as bundle:
                data["request.json"] = bundle.extractfile("request.json").read()
        request = json.loads(data["request.json"])
        assert request["provider_config"]["gpu"] == "L4"
        data["result.json"] = json_bytes(
            failure_result(request, "failed", "compile_error", "test error")
        )
        data["completion.json"] = json_bytes(
            {
                "run_id": request["run_id"],
                "request_hash": request["request_hash"],
                "files": {
                    name: {"size": len(value), "sha256": digest(value)}
                    for name, value in data.items()
                },
            }
        )
        return {"s3_bucket": "example", "s3_key": "bundle"}

    monkeypatch.setattr(BasetenProvider, "api", api)
    monkeypatch.setattr(baseten_backend, "upload_archive", upload)
    monkeypatch.setattr(
        BasetenProvider, "download", staticmethod(lambda entry: data[entry["relative_file_name"]])
    )
    config = ["--config", str(tmp_path / "missing.toml"), "--state-dir", str(tmp_path / "state")]
    manifest = Path(__file__).resolve().parents[1] / "examples/vector-add/task.toml"
    assert main(["submit", str(manifest), "--provider", "baseten", "--gpu", "L4", *config]) == 0
    submitted = json.loads(capsys.readouterr().out)
    assert submitted["handle"]["native_id"] == "job"

    assert main(["result", submitted["run_id"], *config]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execution"]["status"] == "failed"
    assert result["execution"]["error"]["reason"] == "compile_error"
    assert result["execution"]["provider"] == "baseten"
