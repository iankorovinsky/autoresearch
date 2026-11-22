import base64
import json
import shlex
import subprocess
from pathlib import Path

from kernel_compute.common import atomic_json, digest
from kernel_compute.models import JobHandle, Observation, WaterlooConfig
from kernel_compute.providers import environment_lock
from kernel_compute.providers.base import AmbiguousSubmission, InvalidArtifact, ProviderError


class WaterlooProvider:
    id = "waterloo-datasci"

    def __init__(self, config: WaterlooConfig):
        self.config = config

    def snapshot(self):
        return self.config.model_dump()

    def describe(self):
        return {
            "id": self.id,
            "name": "Waterloo Data Science Cluster",
            "queue": "Slurm",
            "node": self.config.node,
            "single_gpu": True,
            "mig_selection": False,
            "logs": "live",
            "cancel": True,
            "profiling": "cuda-event-warm-v1",
        }

    def rpc(self, operation, payload, data=None):
        source = Path(__file__).with_name("remote.py").read_text()
        config = self.config
        target = f"{config.user}@{config.host}" if config.user else config.host
        command = shlex.join(
            ["python3", "-c", source, json.dumps([self.snapshot(), operation, payload])]
        )
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            f"ConnectTimeout={config.connect_timeout_s}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            target,
            command,
        ]
        try:
            proc = subprocess.run(args, input=data or b"", capture_output=True, timeout=90)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderError(f"SSH {operation} transport failed: {exc}") from exc
        if proc.returncode:
            raise ProviderError(
                f"SSH {operation} failed: {proc.stderr.decode(errors='replace')[-4096:]}"
            )
        try:
            response = json.loads(proc.stdout)
        except ValueError as exc:
            raise ProviderError("invalid SSH helper response") from exc
        if not response.get("ok"):
            if operation == "fetch" and response.get("kind") in {
                "ValueError",
                "KeyError",
                "TypeError",
                "JSONDecodeError",
            }:
                raise InvalidArtifact(response.get("error", "invalid remote artifact"))
            raise ProviderError(response.get("error", "remote operation failed"))
        return response["value"]

    def doctor(self):
        return self.rpc("doctor", {"lock_hash": digest(environment_lock())})

    def setup(self):
        lock = environment_lock()
        return self.rpc(
            "setup",
            {
                "lock_hash": digest(lock),
                "lock": base64.b64encode(lock).decode(),
                "bootstrap": Path(__file__).with_name("bootstrap.py").read_text(),
            },
        )

    def setup_status(self):
        return self.rpc("setup_status", {"lock_hash": digest(environment_lock())})

    def preflight(self, request):
        task = request["task"]
        if task["environment"] != self.config.environment:
            raise ProviderError(f"unsupported environment: {task['environment']}")
        if task["resources"]["mig"] == "require":
            raise ProviderError(
                "the Waterloo v1 profile selects full GPUs; MIG selection is not implemented"
            )
        status = self.doctor()
        if not status["node"]:
            raise ProviderError("configured node is not visible in Slurm")
        if not status["environment_ready"]:
            raise ProviderError("worker environment is not ready; run kcompute setup --wait")
        if status["environment"]["lock_hash"] != request["environment_lock_hash"]:
            raise ProviderError("environment hash mismatch")
        if status["environment"]["cuda_home"] != self.config.cuda_home:
            raise ProviderError("profile CUDA path differs from the validated environment")
        return status

    def submit(self, request, bundle):
        self.rpc(
            "stage",
            {"run_id": request["run_id"], "bundle_hash": digest(bundle.read_bytes())},
            bundle.read_bytes(),
        )
        try:
            value = self.rpc(
                "submit", {"run_id": request["run_id"], "request_hash": request["request_hash"]}
            )
        except ProviderError as exc:
            raise AmbiguousSubmission(str(exc)) from exc
        if value.get("ambiguous"):
            raise AmbiguousSubmission(value["message"])
        return JobHandle.model_validate(value["handle"])

    def observe(self, request):
        value = self.rpc("observe", {"run_id": request["run_id"]})
        handle = JobHandle.model_validate(value["handle"]) if value.get("handle") else None
        return handle, Observation.model_validate(value["observation"])

    def read_logs(self, request, cursor=None):
        return self.rpc("logs", {"run_id": request["run_id"], "cursor": cursor})

    def cancel(self, request):
        return self.rpc("cancel", {"run_id": request["run_id"]})

    def fetch(self, request, destination):
        destination.mkdir(parents=True, exist_ok=True)
        allowed = {"request.json", "result.json", "samples.json", "stdout.log", "stderr.log"}
        have = {
            name: digest((destination / name).read_bytes())
            for name in allowed
            if (destination / name).exists()
        }
        value = self.rpc("fetch", {"run_id": request["run_id"], "have": have})
        if not value["complete"]:
            return None
        manifest = value["manifest"]
        if (
            manifest.get("run_id") != request["run_id"]
            or manifest.get("request_hash") != request["request_hash"]
        ):
            raise InvalidArtifact("artifact manifest identity mismatch")
        if not {"request.json", "result.json"}.issubset(manifest["files"]):
            raise InvalidArtifact("artifact manifest is incomplete")
        for name, entry in manifest["files"].items():
            if name not in allowed or not 0 <= entry["size"] <= 4 * 1024**2:
                raise InvalidArtifact("unsafe artifact")
            if name in value["files"]:
                data = base64.b64decode(value["files"][name], validate=True)
            else:
                data = (destination / name).read_bytes()
            if len(data) != entry["size"] or digest(data) != entry["sha256"]:
                raise InvalidArtifact("downloaded artifact checksum mismatch")
            temporary = destination / (name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination / name)
        atomic_json(destination / "completion.json", manifest)
        return json.loads((destination / "result.json").read_text())
