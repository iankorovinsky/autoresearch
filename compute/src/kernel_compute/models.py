"""Versioned coordinator contracts; no provider or GPU imports."""

import math
import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"not a safe relative path: {value!r}")
    if any(part.startswith(".") for part in path.parts) or str(path) != value:
        raise ValueError(f"hidden or noncanonical bundle path: {value!r}")
    if any(part in {"node_modules", "venv", "__pycache__"} for part in path.parts):
        raise ValueError(f"cache/environment path forbidden: {value!r}")
    if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
        raise ValueError("credential files must not be bundled")
    return value


class Resources(Model):
    gpu_count: Literal[1] = 1
    min_gpu_memory_gib: float = Field(default=0, ge=0)
    mig: Literal["forbid", "allow", "require"] = "forbid"
    cpu_cores: int = Field(default=4, ge=1, le=256)
    host_memory_gib: int = Field(default=8, ge=1, le=4096)
    wall_time_s: int = Field(default=600, ge=1, le=604800)


class Measurement(Model):
    warmup_iterations: int = Field(default=10, ge=1, le=1000)
    sample_count: int = Field(default=100, ge=2, le=10000)
    cache_policy: Literal["warm"] = "warm"
    timing_scope: Literal["callable_current_stream"] = "callable_current_stream"


class Validation(Model):
    mode: Literal["none", "reference"] = "none"
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=1, max_length=100)
    rtol: float | None = Field(default=None, ge=0)
    atol: float | None = Field(default=None, ge=0)
    equal_nan: bool = False

    @model_validator(mode="after")
    def explicit_tolerances(self):
        if self.mode == "reference" and (self.rtol is None or self.atol is None):
            raise ValueError("reference validation requires explicit rtol and atol")
        return self


class Task(Model):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=128)
    entrypoint: str
    files: list[str] = Field(min_length=1, max_length=500)
    environment: str = "cuda-pytorch-v1"
    case: dict[str, Any] = Field(default_factory=dict)
    resources: Resources = Field(default_factory=Resources)
    measurement: Measurement = Field(default_factory=Measurement)
    validation: Validation = Field(default_factory=Validation)

    @field_validator("files")
    @classmethod
    def safe_files(cls, files):
        if len(files) != len(set(files)):
            raise ValueError("duplicate bundle paths")
        return [safe_relative(name) for name in files]

    @model_validator(mode="after")
    def valid_entrypoint(self):
        path, sep, function = self.entrypoint.partition(":")
        if not sep or not function.isidentifier() or not path.endswith(".py"):
            raise ValueError("entrypoint must be file.py:function")
        if safe_relative(path) not in self.files:
            raise ValueError("entrypoint must be in files")
        return self


class WaterlooConfig(Model):
    host: str = "datasci-gpu.cs.uwaterloo.ca"
    user: str | None = None
    remote_root: str = ".cache/kcompute"
    node: str = "gpu-8-vm"
    partition: str = "ALL"
    cuda_home: str = "/usr/local/cuda-13.2"
    environment: str = "cuda-pytorch-v1"
    connect_timeout_s: int = Field(default=15, ge=1, le=120)

    @field_validator("host", "user", "node", "partition")
    @classmethod
    def safe_identifier(cls, value):
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
            raise ValueError("invalid host/user/node/partition identifier")
        return value

    @field_validator("remote_root")
    @classmethod
    def safe_root(cls, value):
        # Keep the run root under HOME, without permitting shell expansion.
        if not re.fullmatch(r"[a-zA-Z0-9_. /-]+", value) or value.startswith("/"):
            raise ValueError("remote_root must be relative to the remote home directory")
        if ".." in PurePosixPath(value).parts or value in {"", "."}:
            raise ValueError("invalid remote_root")
        return value

    @field_validator("cuda_home")
    @classmethod
    def safe_cuda(cls, value):
        if not re.fullmatch(r"/[a-zA-Z0-9_./-]+", value):
            raise ValueError("cuda_home must be an absolute path")
        return value


class JobHandle(Model):
    provider: str
    run_id: str
    native_id: str
    cluster: str
    submitted_at: str
    artifact_locator: str


class Observation(Model):
    state: str
    raw_state: str | None = None
    reason: str | None = None
    exit_code: str | None = None
    timestamps: dict[str, str] = Field(default_factory=dict)


def validate_result(result: dict, request: dict) -> dict:
    """Validate identity and metrics even for results retrieved from a provider."""
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    for group in ("execution", "validation", "environment", "unavailable_reasons"):
        if not isinstance(result.get(group), dict):
            raise ValueError(f"missing or malformed result group: {group}")
    for group in ("latency", "memory"):
        if group not in result or (
            result[group] is not None and not isinstance(result[group], dict)
        ):
            raise ValueError(f"missing or malformed result group: {group}")
    for key in ("run_id", "request_hash", "task_hash"):
        if result.get(key) != request[key]:
            raise ValueError(f"result {key} mismatch")
    if result.get("schema_version") != 1:
        raise ValueError("unsupported result schema")
    if result.get("execution", {}).get("status") not in {
        "succeeded",
        "failed",
        "timed_out",
        "cancelled",
    }:
        raise ValueError("invalid result status")
    validation = result.get("validation", {}).get("status")
    if validation not in {"passed", "failed", "not_requested", "not_run"}:
        raise ValueError("invalid validation status")
    if result["execution"]["status"] == "succeeded":
        expected = (
            "passed" if request["task"]["validation"]["mode"] == "reference" else "not_requested"
        )
        if validation != expected or not result.get("latency"):
            raise ValueError("success requires measurements and the requested validation")
    latency = result.get("latency")
    if latency:
        if latency.get("units") != "us":
            raise ValueError("latency must use microseconds")
        for key in ("mean", "median", "min", "max", "stddev", "p95"):
            value = latency.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid latency {key}")
        if latency.get("sample_count") != request["task"]["measurement"]["sample_count"]:
            raise ValueError("sample count mismatch")
        settings = request["task"]["measurement"]
        for key in ("cache_policy", "timing_scope", "warmup_iterations"):
            if latency.get(key) != settings[key]:
                raise ValueError(f"measurement protocol mismatch: {key}")
    memory = result.get("memory")
    if memory:
        if (
            memory.get("units") != "bytes"
            or memory.get("scope") != "pytorch_allocator_including_resident_inputs_and_model"
        ):
            raise ValueError("invalid memory measurement scope/units")
        for key in ("baseline_allocated", "baseline_reserved", "peak_allocated", "peak_reserved"):
            if type(memory.get(key)) is not int or memory[key] < 0:
                raise ValueError(f"invalid memory {key}")
    return result
