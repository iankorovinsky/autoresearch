"""Run frozen kernel bundles and GPU probes as Baseten Training Jobs."""

import json
import os
import re
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import field_validator

from kernel_compute.common import TERMINAL, atomic_json, digest
from kernel_compute.models import JobHandle, Model, Observation
from kernel_compute.providers import WORKER_LOCK, environment_lock
from kernel_compute.providers.base import (
    AmbiguousSubmission,
    InvalidArtifact,
    ProviderError,
    RejectedSubmission,
)

API = "https://api.baseten.co/v1"
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
GPU_TYPES = ("T4", "L4", "A10G", "A100", "H100MIG", "H100", "H200", "B200", "RTX-PRO-6000")
# Additional NVIDIA accelerator selectors exposed by Baseten's Truss SDK.
EXTRA_TYPES = ("V100", "A100_40GB", "L40S", "B300", "GB300")
NVIDIA_TYPES = GPU_TYPES + EXTRA_TYPES
GPU_SELECTOR = re.compile(r"[A-Z0-9][A-Z0-9_-]{0,63}")
GPU_ALIASES = {
    "H100_40GB": "H100MIG",
    "H100-40GB": "H100MIG",
    "H100-MIG": "H100MIG",
    "RTX_PRO_6000": "RTX-PRO-6000",
}
ACCELERATORS = {"H100MIG": "H100_40GB", "RTX-PRO-6000": "RTX_PRO_6000"}
TOOLS = ["nvidia-smi", "cuda-events", "nvcc", "torch-profiler", "triton", "nsys", "ncu"]
FILES = {"request.json", "result.json", "samples.json", "stdout.log", "stderr.log"}
MAX_ARTIFACT = 4 * 1024**2
PROBE_WAIT_SECONDS = 1200
STATES = {
    "PENDING": "queued",
    "CREATED": "queued",
    "DEPLOYING": "queued",
    "RUNNING": "running",
    "COMPLETED": "succeeded",
    "DEPLOY_FAILED": "failed",
    "FAILED": "failed",
    "PREEMPTED": "failed",
    "STOPPED": "cancelled",
}


def gpu_name(value):
    value = value.strip().upper()
    return GPU_ALIASES.get(value, value)


class BasetenConfig(Model):
    gpu: str = "A10G"
    project_name: str = "kcompute"
    project_id: str | None = None

    @field_validator("gpu")
    @classmethod
    def supported_gpu(cls, value):
        gpu = gpu_name(value)
        if not GPU_SELECTOR.fullmatch(gpu):
            raise ValueError(f"invalid Baseten GPU {value!r}; see providers list")
        return gpu

    @field_validator("project_name", "project_id")
    @classmethod
    def valid_project(cls, value):
        if value is not None and (not value.strip() or len(value) > 128):
            raise ValueError("Baseten project names and IDs must contain 1–128 characters")
        return value


class BasetenHTTPError(ProviderError):
    def __init__(self, status, detail):
        self.status = status
        super().__init__(f"Baseten HTTP {status}: {detail}")


def api_key():
    return os.environ.get("BASETEN_API_KEY", "").strip()


def upload_archive(archive, credentials):
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise ProviderError(
            "Install Baseten upload support with 'uv sync --extra baseten'."
        ) from exc
    try:
        client = boto3.client("s3", **credentials["creds"])
        client.upload_file(str(archive), credentials["s3_bucket"], credentials["s3_key"])
    except (BotoCoreError, ClientError, OSError) as exc:
        raise ProviderError(f"Baseten source upload failed ({type(exc).__name__})") from exc
    return {key: credentials[key] for key in ("s3_bucket", "s3_key")}


class BasetenProvider:
    id = "baseten"

    def __init__(self, config: BasetenConfig):
        self.config = config

    def snapshot(self):
        return self.config.model_dump()

    def api(self, path, *, body=None):
        key = api_key()
        if not key:
            raise ProviderError("Set BASETEN_API_KEY to a Baseten workspace API key.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        request = Request(
            API + path, headers=headers, data=None if body is None else json.dumps(body).encode()
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                return payload
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read(4096))
                detail = payload.get("error") or payload.get("detail") or payload.get("message")
            except (AttributeError, UnicodeError, ValueError):
                detail = None
            detail = str(detail or exc.reason).replace(key, "[redacted]")[:1000]
            raise BasetenHTTPError(exc.code, detail) from exc
        except URLError as exc:
            raise ProviderError(f"Could not reach Baseten ({type(exc.reason).__name__})") from exc
        except (OSError, ValueError) as exc:
            raise ProviderError(f"Baseten response unavailable ({type(exc).__name__})") from exc

    def describe(self):
        result = {
            "id": self.id,
            "name": "Baseten",
            "queue": "Baseten Training Jobs",
            "gpu": self.config.gpu,
            "gpu_types": list(NVIDIA_TYPES),
            "cheapest_published_gpu": {"gpu": "T4", "usd_per_minute": 0.01052},
            "single_gpu": True,
            "mig_selection": True,
            "cancel": True,
            "logs": "after_completion",
            "profiling": "cuda-event-warm-v1",
            "probe_tools": TOOLS,
        }
        if api_key():
            try:
                instances = self.workspace_instances()
                result["workspace_gpu_instances"] = instances
                workspace_types = {gpu_name(row["gpu_type"]) for row in instances}
                result["gpu_types"] += sorted(
                    gpu
                    for gpu in workspace_types - set(NVIDIA_TYPES)
                    if GPU_SELECTOR.fullmatch(gpu)
                )
            except ProviderError as exc:
                result["workspace_catalog_error"] = str(exc)
        return result

    def workspace_instances(self):
        rows = self.api("/instance_type_prices")["instance_types"]
        return [
            {**row["instance_type"], "usd_per_minute": row["price"]}
            for row in rows
            if row["instance_type"]["gpu_count"] > 0 and row["instance_type"]["gpu_type"]
        ]

    def project(self, *, create=False):
        if self.config.project_id:
            project_id = quote(self.config.project_id, safe="")
            return self.api(f"/training_projects/{project_id}")["training_project"]
        name = self.config.project_name
        query = urlencode({"name": name})
        projects = self.api(f"/training_projects?{query}")["training_projects"]
        matches = [project for project in projects if project["name"] == name]
        if len(matches) > 1:
            raise ProviderError("Multiple Baseten projects share this name. Configure project_id.")
        if matches or not create:
            return next(iter(matches), None)
        return self.api("/training_projects", body={"training_project": {"name": name}})[
            "training_project"
        ]

    def doctor(self):
        project = self.project()
        return {"ready": project is not None, "gpu": self.config.gpu, "project": project}

    def setup(self):
        return {"ready": True, "gpu": self.config.gpu, "project": self.project(create=True)}

    def preflight(self, request):
        task = request["task"]
        mig = task["resources"]["mig"]
        selected_mig = self.config.gpu == "H100MIG"
        if (mig == "require" and not selected_mig) or (mig == "forbid" and selected_mig):
            raise ProviderError("The selected Baseten GPU conflicts with the task's MIG setting.")
        if task["environment"] != "cuda-pytorch-v1":
            raise ProviderError(f"unsupported environment: {task['environment']}")
        if request["environment_lock_hash"] != digest(environment_lock()):
            raise ProviderError("worker environment lock changed since the batch was prepared")
        if not api_key():
            raise ProviderError("Set BASETEN_API_KEY to a Baseten workspace API key.")

    @staticmethod
    def jobs_path(project_id):
        return f"/training_projects/{quote(project_id, safe='')}/jobs"

    def launch(self, name, cpu, memory, bundle=None):
        mode = "run" if bundle is not None else "probe"
        try:
            if self.config.gpu not in NVIDIA_TYPES and not any(
                gpu_name(row["gpu_type"]) == self.config.gpu for row in self.workspace_instances()
            ):
                raise ProviderError(f"Unknown Baseten GPU {self.config.gpu!r}; see providers list.")
            project = self.project(create=True)
            credentials = self.api("/blobs/credentials/train")
            with tempfile.TemporaryDirectory(prefix="kcompute-baseten-") as temp:
                archive = Path(temp) / "source.tar"
                providers = Path(__file__).parent
                with tarfile.open(archive, "w") as source:
                    source.add(providers / "baseten_runner.py", arcname="runner.py")
                    source.add(WORKER_LOCK, arcname="worker.lock")
                    if bundle is None:
                        source.add(providers / "modal_probe.py", arcname="probe.py")
                    else:
                        source.add(bundle, arcname="bundle.tar")
                artifact = upload_archive(archive, credentials)
        except (KeyError, ProviderError, OSError) as exc:
            raise RejectedSubmission(str(exc)) from exc

        accelerator = ACCELERATORS.get(self.config.gpu, self.config.gpu)
        # A fresh container installs the same hash-locked worker used on other providers.
        # Nsight installation is optional so the runtime probe can report missing tools.
        commands = [
            "timeout -k 10s 900s bash -ec "
            "'apt-get update && apt-get install -y python3 python3-dev python3-venv "
            "&& python3 -m venv /opt/kcompute-venv "
            "&& /opt/kcompute-venv/bin/python -m pip install --require-hashes --only-binary=:all: "
            "--no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 -r worker.lock'",
        ]
        if bundle is None:
            commands.append("timeout -k 5s 90s apt-get install -y cuda-nsight-systems-13-0 || true")
            commands.append("timeout -k 5s 90s apt-get install -y cuda-nsight-compute-13-0 || true")
        commands.append(f"/opt/kcompute-venv/bin/python runner.py {mode}")
        job = {
            "name": name,
            "image": {"base_image": CUDA_IMAGE},
            "compute": {
                "node_count": 1,
                "cpu_count": cpu,
                "memory": f"{memory}Gi",
                "accelerator": {"accelerator": accelerator, "count": 1},
            },
            "runtime": {
                "start_commands": commands,
                "artifacts": [artifact],
                "cache_config": {"enabled": False},
                "checkpointing_config": {"enabled": True},
                "environment_variables": {
                    "CUDA_HOME": "/usr/local/cuda",
                    "DEBIAN_FRONTEND": "noninteractive",
                },
            },
            "enable_baseten_workdir": True,
        }
        path = self.jobs_path(project["id"])
        try:
            accepted = self.api(path, body={"training_job": job})["training_job"]
            if (
                not isinstance(accepted, dict)
                or not {"id", "training_project_id", "created_at", "current_status"}
                <= accepted.keys()
            ):
                raise ProviderError("Baseten returned an incomplete job receipt")
            return accepted
        except BasetenHTTPError as exc:
            error = (
                RejectedSubmission
                if exc.status in {400, 401, 403, 404, 422, 429}
                else AmbiguousSubmission
            )
            raise error(str(exc)) from exc
        except (KeyError, ProviderError) as exc:
            raise AmbiguousSubmission(
                f"Baseten acknowledgement unavailable for {name}: {exc}"
            ) from exc

    def submit(self, request, bundle):
        resources = request["task"]["resources"]
        job = self.launch(
            request["run_id"], resources["cpu_cores"], resources["host_memory_gib"], bundle
        )
        return self.handle(request, job)

    def handle(self, request, job):
        return JobHandle(
            provider=self.id,
            run_id=request["run_id"],
            native_id=job["id"],
            cluster=job["training_project_id"],
            submitted_at=job["created_at"],
            artifact_locator=self.job_path(job) + "/checkpoint_files",
        )

    def find_job(self, name):
        project = self.project()
        if project is None:
            return None
        jobs = self.api(self.jobs_path(project["id"]))["training_jobs"]
        matches = [job for job in jobs if job["name"] == name]
        if len(matches) > 1:
            raise ProviderError(
                f"Multiple Baseten jobs match {name}. Check the training dashboard."
            )
        return next(iter(matches), None)

    def job_path(self, job):
        return self.jobs_path(job["training_project_id"]) + "/" + quote(job["id"], safe="")

    @staticmethod
    def state(job):
        return STATES.get(job["current_status"].removeprefix("TRAINING_JOB_"), "unknown")

    def observe(self, request):
        job = self.find_job(request["run_id"])
        if job is None:
            return None, Observation(
                state="unknown", reason="Baseten Training Job could not be found"
            )
        return self.handle(request, job), Observation(
            state=self.state(job), raw_state=job["current_status"], reason=job.get("error_message")
        )

    def stop(self, job):
        if self.state(job) in TERMINAL:
            return {"requested": False, "state": self.state(job)}
        self.api(self.job_path(job) + "/stop", body={})
        return {"requested": True, "state": "cancelling"}

    def cancel(self, request):
        job = self.find_job(request["run_id"])
        return self.stop(job) if job else {"requested": False, "state": "unknown"}

    def checkpoint_files(self, job):
        try:
            response = self.api(self.job_path(job) + "/checkpoint_files")
        except BasetenHTTPError as exc:
            if exc.status == 404:
                return {}
            raise
        return {
            entry["relative_file_name"]: entry
            for entry in response["presigned_urls"]
            if entry["node_rank"] == 0
        }

    @staticmethod
    def download(entry):
        if entry["size_bytes"] > MAX_ARTIFACT:
            raise InvalidArtifact("oversized Baseten artifact")
        url = entry["url"]
        if urlsplit(url).scheme != "https":
            raise InvalidArtifact("invalid Baseten artifact URL")
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read(MAX_ARTIFACT + 1)
        except (URLError, OSError) as exc:
            raise ProviderError("Could not download Baseten checkpoint file") from exc
        if len(data) != entry["size_bytes"] or len(data) > MAX_ARTIFACT:
            raise InvalidArtifact("Baseten artifact size mismatch")
        return data

    def fetch(self, request, destination):
        job = self.find_job(request["run_id"])
        if job is None or self.state(job) != "succeeded":
            return None
        entries = self.checkpoint_files(job)
        if "completion.json" not in entries:
            return None
        manifest = json.loads(self.download(entries["completion.json"]))
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != request["run_id"]
            or manifest.get("request_hash") != request["request_hash"]
        ):
            raise InvalidArtifact("Baseten artifact manifest identity mismatch")
        files = manifest.get("files")
        if (
            not isinstance(files, dict)
            or not {"request.json", "result.json"} <= files.keys()
            or files.keys() - FILES
        ):
            raise InvalidArtifact("invalid Baseten artifact manifest")
        if files.keys() - entries.keys():
            return None
        destination.mkdir(parents=True, exist_ok=True)
        for name, entry in files.items():
            data = self.download(entries[name])
            if (
                not isinstance(entry, dict)
                or len(data) != entry.get("size")
                or digest(data) != entry.get("sha256")
            ):
                raise InvalidArtifact(f"Baseten artifact checksum mismatch: {name}")
            (destination / name).write_bytes(data)
        atomic_json(destination / "completion.json", manifest)
        return json.loads((destination / "result.json").read_bytes())

    def read_logs(self, request, cursor=None):
        job = self.find_job(request["run_id"])
        entries = self.checkpoint_files(job) if job and self.state(job) in TERMINAL else {}
        chunks, offsets = {}, {}
        for stream in ("stdout", "stderr"):
            entry = entries.get(f"{stream}.log")
            offset = (cursor or {}).get(stream, 0)
            data = self.download(entry) if entry else b""
            chunk = data[offset : offset + 65536]
            chunks[stream] = chunk.decode(errors="replace")
            offsets[stream] = offset + len(chunk)
        return {"chunks": chunks, "cursor": offsets}

    def probe(self):
        name = f"probe-{uuid4().hex}"
        job = None
        try:
            job = self.launch(name, 4, 16)
            path = self.job_path(job)
            deadline = time.monotonic() + PROBE_WAIT_SECONDS
            while time.monotonic() < deadline:
                state = self.state(job)
                if state in TERMINAL and state != "succeeded":
                    reason = job.get("error_message") or job["current_status"]
                    raise ProviderError(f"Baseten probe {job['id']} failed: {reason}")
                if state == "succeeded":
                    entry = self.checkpoint_files(job).get("probe.json")
                    if entry:
                        report = json.loads(self.download(entry))
                        return {
                            "provider": self.id,
                            "requested_gpu": self.config.gpu,
                            "job_id": job["id"],
                            **report,
                        }
                time.sleep(5)
                job = self.api(path)["training_job"]
            raise TimeoutError(f"Baseten probe {job['id']} exceeded {PROBE_WAIT_SECONDS}s")
        except RejectedSubmission:
            raise
        except (KeyboardInterrupt, TimeoutError, ProviderError):
            # If a launch acknowledgement was lost, recover by the unique name before stopping.
            try:
                if job is None and api_key():
                    job = self.find_job(name)
                if job is not None:
                    self.stop(job)
            except ProviderError as exc:
                identity = job["id"] if job else name
                raise ProviderError(f"Could not stop Baseten probe {identity}: {exc}") from exc
            raise
