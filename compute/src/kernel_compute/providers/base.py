from pathlib import Path
from typing import Protocol

from kernel_compute.models import JobHandle, Observation


class ProviderError(RuntimeError):
    pass


class AmbiguousSubmission(ProviderError):
    """Acceptance is unknown; never automatically launch a replacement."""


class InvalidArtifact(ValueError):
    """A finalized artifact violates the protocol; retrying execution is unsafe."""


class ComputeProvider(Protocol):
    id: str

    def describe(self) -> dict: ...
    def snapshot(self) -> dict: ...
    def preflight(self, request: dict) -> dict: ...
    def submit(self, request: dict, bundle: Path) -> JobHandle: ...
    def observe(self, request: dict) -> tuple[JobHandle | None, Observation]: ...
    def read_logs(self, request: dict, cursor: dict | None) -> dict: ...
    def cancel(self, request: dict) -> dict: ...
    def fetch(self, request: dict, destination: Path) -> dict | None: ...
