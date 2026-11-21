"""Select runtimes without modifying shell/global installations."""

import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def clean_environment(component: Path | None = None):
    env = os.environ.copy()
    for key in (
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "UV_PROJECT_ENVIRONMENT",
        "UV_NO_SYNC",
        "UV_FROZEN",
        "UV_ACTIVE",
        "NODE_PATH",
        "NODE_OPTIONS",
        "KCOMPUTE_LIVE",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    if component is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(component / ".venv")
    return env


def stop(process, first=signal.SIGTERM):
    # The leader may already have exited while descendants still hold pipes/ports.
    for sig in (first, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.poll()
            return
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
    process.poll()


def capture(args, *, cwd=None, env=None, timeout=30):
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env or clean_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        stop(process)
        process.communicate()
        raise


def version_tuple(value):
    match = re.search(r"\b[vV]?(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError("unrecognized version")
    return tuple(map(int, match.groups()))


@dataclass(frozen=True)
class NodeRuntime:
    node: Path
    npm: Path
    version: tuple[int, int, int]
    source: str

    def environment(self):
        env = clean_environment()
        env["PATH"] = str(self.node.parent) + os.pathsep + env.get("PATH", "")
        return env


def select_node(root: Path) -> NodeRuntime:
    override = os.environ.get("REPO_DEV_NODE_BIN")
    candidates = []
    if override:
        candidates.append((Path(override).expanduser().resolve() / "node", "REPO_DEV_NODE_BIN"))
    else:
        # Prefer the repo's recommended major when installed through nvm.
        nvm = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm")) / "versions/node"
        versions = []
        for path in nvm.glob("v24.*/bin/node"):
            try:
                versions.append((version_tuple(path.parent.parent.name), path))
            except ValueError:
                pass
        candidates.extend(
            (path, "installed Node 24 (nvm)") for _, path in sorted(versions, reverse=True)
        )
        active = shutil.which("node")
        if active:
            # Keep the installation's bin path, rather than resolving symlinks away from npm.
            candidates.append((Path(active).absolute(), "PATH"))
    for node, source in candidates:
        npm = node.parent / "npm"
        if not node.is_file() or not npm.is_file():
            continue
        try:
            proc = capture([str(node), "--version"], timeout=5)
            version = version_tuple(proc.stdout)
            if proc.returncode == 0 and version >= (22, 12, 0):
                return NodeRuntime(node, npm, version, source)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    raise ValueError(
        "Node >=22.12 with its matching npm is required; install Node 24 or set REPO_DEV_NODE_BIN to its bin directory"
    )


def uv_executable():
    path = shutil.which("uv")
    if path is None:
        raise ValueError("uv is missing; install it before using Python component tasks")
    return path
