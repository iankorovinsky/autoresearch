"""Modal Sandbox scheduling and Volume-backed task results."""

import io
import json
import re
from pathlib import Path

import modal
from modal.exception import (
    AuthError,
    Error,
    ImageBuildError,
    InvalidError,
    NotFoundError,
    PermissionDeniedError,
    SandboxTerminatedError,
)
from pydantic import field_validator

from kernel_compute.common import atomic_json, digest, json_bytes, now
from kernel_compute.models import JobHandle, Model, Observation
from kernel_compute.providers import WORKER_LOCK, environment_lock
from kernel_compute.providers.base import (
    AmbiguousSubmission,
    InvalidArtifact,
    ProviderError,
    RejectedSubmission,
)

GPU_TYPES = (
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "RTX-PRO-6000",
    "H100",
    "H100!",
    "H200",
    "B200",
    "B200+",
    "B300",
)
TOOLS = ["nvidia-smi", "cuda-events", "nvcc", "torch-profiler", "triton", "nsys", "ncu"]
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
CUDA_HOME = "/usr/local/cuda"
PYTHON = "/opt/kcompute-venv/bin/python"
FILES = {"request.json", "result.json", "samples.json", "stdout.log", "stderr.log"}
MAX_ARTIFACT = 4 * 1024**2


class ModalConfig(Model):
    gpu: str = "T4"
    environment_name: str | None = None
    app_name: str = "kcompute"
    volume_name: str = "kcompute-artifacts-v2"

    @field_validator("gpu")
    @classmethod
    def supported_gpu(cls, value):
        gpu = value.upper()
        if gpu not in GPU_TYPES:
            raise ValueError(f"unknown Modal GPU {value!r}; choose from {', '.join(GPU_TYPES)}")
        return gpu

    @field_validator("environment_name", "app_name", "volume_name")
    @classmethod
    def valid_name(cls, value):
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", value):
            raise ValueError("invalid Modal environment, app or volume name")
        return value


def worker_image():
    # The inventory and lock hash are generated in the image that will run the worker.
    inventory = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({'packages': {d.metadata['Name']: d.version for d in m.distributions()}, "
        f"'lock_hash': '{digest(environment_lock())}'}}))"
    )
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .apt_install("cuda-nsight-systems-13-0")
        .add_local_file(str(WORKER_LOCK), "/opt/kcompute-worker.lock", copy=True)
        .run_commands(
            "python -m venv /opt/kcompute-venv",
            f"{PYTHON} -m pip install --require-hashes --only-binary=:all: --no-cache-dir "
            "--extra-index-url https://download.pytorch.org/whl/cu130 -r /opt/kcompute-worker.lock",
            f"""{PYTHON} -c "{inventory}" > /opt/kcompute-venv/ready.json""",
        )
        .env({"CUDA_HOME": CUDA_HOME})
    )


def modal_error(exc):
    if isinstance(exc, AuthError):
        return ProviderError("Modal authentication required. Run 'uv run modal token new'.")
    return ProviderError(f"Modal {type(exc).__name__}: {exc}")


class ModalProvider:
    id = "modal"

    def __init__(self, config: ModalConfig):
        self.config = config

    def snapshot(self):
        return self.config.model_dump()

    def describe(self):
        return {
            "id": self.id,
            "name": "Modal",
            "queue": "Modal Sandbox",
            "gpu": self.config.gpu,
            "gpu_types": list(GPU_TYPES),
            "single_gpu": True,
            "mig_selection": False,
            "logs": "after_completion",
            "cancel": True,
            "profiling": "cuda-event-warm-v1",
            "probe_tools": TOOLS,
        }

    def app(self, create=False):
        return modal.App.lookup(
            self.config.app_name,
            environment_name=self.config.environment_name,
            create_if_missing=create,
        )

    def volume(self, create=False):
        return modal.Volume.from_name(
            self.config.volume_name,
            environment_name=self.config.environment_name,
            create_if_missing=create,
            version=2 if create else None,
        )

    def doctor(self):
        try:
            modal.Client.from_env()
            try:
                self.app()
                self.volume().hydrate()
                configured = True
            except NotFoundError:
                configured = False
        except Error as exc:
            raise modal_error(exc) from exc
        return {"ready": configured, "authenticated": True, "gpu": self.config.gpu}

    def setup(self):
        try:
            app = self.app(create=True)
            self.volume(create=True).hydrate()
            image = worker_image()
            sandbox = modal.Sandbox.create(
                PYTHON,
                "-c",
                "import torch; print(torch.__version__)",
                app=app,
                image=image,
                cpu=1,
                memory=2048,
                timeout=90,
                environment_name=self.config.environment_name,
            )
            sandbox.wait()
            if sandbox.returncode != 0:
                raise ProviderError(f"Modal setup failed: {sandbox.stderr.read()[-4096:]}")
        except Error as exc:
            raise modal_error(exc) from exc
        return {"ready": True, "torch_version": sandbox.stdout.read().strip().rsplit("\n", 1)[-1]}

    def preflight(self, request):
        task = request["task"]
        if task["environment"] != "cuda-pytorch-v1":
            raise ProviderError(f"unsupported environment: {task['environment']}")
        if task["resources"]["mig"] == "require":
            raise ProviderError("Modal GPU profiles do not select MIG devices")
        if task["resources"]["wall_time_s"] > 86400:
            raise ProviderError("Modal Sandboxes support wall_time_s up to 86400")
        if request["environment_lock_hash"] != digest(environment_lock()):
            raise ProviderError("worker environment lock changed since the batch was prepared")
        try:
            modal.Client.from_env()
        except Error as exc:
            raise modal_error(exc) from exc

    def submit(self, request, bundle):
        run_id = request["run_id"]
        resources = request["task"]["resources"]
        try:
            app = self.app(create=True)
            volume = self.volume(create=True)
            with volume.batch_upload() as upload:
                upload.put_file(bundle, f"/runs/{run_id}/bundle.tar")
            image = worker_image().add_local_file(
                str(Path(__file__).with_name("modal_runner.py")), "/opt/kcompute-runner.py"
            )
        except (Error, OSError, ValueError) as exc:
            raise RejectedSubmission(str(modal_error(exc))) from exc
        try:
            sandbox = modal.Sandbox.create(
                PYTHON,
                "/opt/kcompute-runner.py",
                app=app,
                name=run_id,
                environment_name=self.config.environment_name,
                image=image,
                gpu=self.config.gpu,
                cpu=resources["cpu_cores"],
                memory=resources["host_memory_gib"] * 1024,
                timeout=resources["wall_time_s"],
                volumes={"/kcompute-data": volume.with_mount_options(sub_path=f"/runs/{run_id}")},
            )
        except (AuthError, ImageBuildError, InvalidError, PermissionDeniedError) as exc:
            raise RejectedSubmission(str(modal_error(exc))) from exc
        except Error as exc:
            raise AmbiguousSubmission(str(modal_error(exc))) from exc
        handle = self.handle(request, sandbox)
        try:
            with volume.batch_upload() as upload:
                upload.put_file(
                    io.BytesIO(json_bytes(handle.model_dump())), f"/receipts/{run_id}.json"
                )
        except (Error, OSError) as exc:
            error = modal_error(exc)
            raise AmbiguousSubmission(str(error)) from exc
        return handle

    def handle(self, request, sandbox):
        return JobHandle(
            provider=self.id,
            run_id=request["run_id"],
            native_id=sandbox.object_id,
            cluster=self.config.environment_name or "modal",
            submitted_at=now(),
            artifact_locator=f"{self.config.volume_name}/runs/{request['run_id']}",
        )

    def read(self, path, limit=MAX_ARTIFACT):
        data = bytearray()
        try:
            for chunk in self.volume().read_file(path):
                data.extend(chunk)
                if len(data) > limit:
                    raise InvalidArtifact(f"oversized Modal artifact: {path}")
        except (FileNotFoundError, NotFoundError):
            return None
        except Error as exc:
            raise modal_error(exc) from exc
        return bytes(data)

    def sandbox(self, request):
        run_id = request["run_id"]
        receipt = self.read(f"/receipts/{run_id}.json", limit=8192)
        try:
            if receipt is not None:
                handle = JobHandle.model_validate_json(receipt)
                if handle.run_id != run_id or handle.provider != self.id:
                    raise ProviderError("Modal submission receipt identity mismatch")
                return handle, modal.Sandbox.from_id(handle.native_id)
            sandbox = modal.Sandbox.from_name(
                self.config.app_name, run_id, environment_name=self.config.environment_name
            )
        except NotFoundError:
            return None, None
        except (Error, ValueError) as exc:
            raise modal_error(exc) from exc
        return self.handle(request, sandbox), sandbox

    def observe(self, request):
        try:
            handle, sandbox = self.sandbox(request)
            if sandbox is None:
                if self.completion(request) is not None:
                    return None, Observation(state="succeeded", raw_state="artifacts_available")
                return None, Observation(state="unknown", reason="Modal Sandbox could not be found")
            code = sandbox.poll()
            if code is None:
                # Modal's poll API reports no exit code while provisioning or executing.
                return handle, Observation(state="queued", raw_state="pending_or_running")
            state = {0: "succeeded", 124: "timed_out"}.get(code, "failed")
            if code == 137:
                try:
                    sandbox.wait()
                except SandboxTerminatedError:
                    state = "cancelled"
            return handle, Observation(state=state, raw_state=state, exit_code=str(code))
        except Error as exc:
            raise modal_error(exc) from exc
        except InvalidArtifact as exc:
            raise ProviderError(str(exc)) from exc

    def cancel(self, request):
        try:
            _, sandbox = self.sandbox(request)
            if sandbox is None:
                return {"requested": False, "state": "unknown"}
            if sandbox.poll() is not None:
                return {"requested": False, "state": self.observe(request)[1].state}
            sandbox.terminate()
        except Error as exc:
            raise modal_error(exc) from exc
        return {"requested": True, "state": "cancelling"}

    def completion(self, request):
        return self.read(f"/runs/{request['run_id']}/completion.json")

    def fetch(self, request, destination):
        completion = self.completion(request)
        if completion is None:
            return None
        manifest = json.loads(completion)
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != request["run_id"]
            or manifest.get("request_hash") != request["request_hash"]
        ):
            raise InvalidArtifact("artifact manifest identity mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or not {"request.json", "result.json"} <= files.keys():
            raise InvalidArtifact("artifact manifest is incomplete")
        if files.keys() - FILES:
            raise InvalidArtifact("unexpected Modal artifacts")
        destination.mkdir(parents=True, exist_ok=True)
        for name, entry in files.items():
            data = self.read(f"/runs/{request['run_id']}/{name}")
            if (
                data is None
                or not isinstance(entry, dict)
                or len(data) != entry.get("size")
                or digest(data) != entry.get("sha256")
            ):
                raise InvalidArtifact(f"Modal artifact checksum mismatch: {name}")
            temporary = destination / (name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination / name)
        atomic_json(destination / "completion.json", manifest)
        return json.loads((destination / "result.json").read_bytes())

    def read_logs(self, request, cursor=None):
        cursor = cursor or {}
        chunks, offsets = {}, {}
        for name in ("stdout", "stderr"):
            offset = cursor.get(name, 0)
            data = self.read(f"/runs/{request['run_id']}/{name}.log") or b""
            chunk = data[offset : offset + 65536]
            chunks[name] = chunk.decode(errors="replace")
            offsets[name] = offset + len(chunk)
        return {"chunks": chunks, "cursor": offsets}

    def probe(self):
        try:
            app = self.app(create=True)
            image = worker_image().add_local_file(
                str(Path(__file__).with_name("modal_probe.py")), "/opt/kcompute-probe.py"
            )
            sandbox = modal.Sandbox.create(
                PYTHON,
                "/opt/kcompute-probe.py",
                app=app,
                image=image,
                gpu=self.config.gpu,
                cpu=2,
                memory=4096,
                timeout=300,
                environment_name=self.config.environment_name,
            )
            try:
                sandbox.wait()
                if sandbox.returncode != 0:
                    raise ProviderError(f"Modal probe failed: {sandbox.stderr.read()[-4096:]}")
                # The NVIDIA image prints its startup banner before the JSON report.
                report = json.loads(sandbox.stdout.read().strip().rsplit("\n", 1)[-1])
                diagnostics = sandbox.stderr.read().strip()
                if diagnostics:
                    report["diagnostics"] = diagnostics[-4000:]
                return {"provider": self.id, "requested_gpu": self.config.gpu, **report}
            except KeyboardInterrupt:
                sandbox.terminate()
                raise
        except Error as exc:
            raise modal_error(exc) from exc
