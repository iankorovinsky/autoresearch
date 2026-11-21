"""Read-only local diagnostics. Secret values and subprocess output are never reported."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from repo_dev.runtime import capture, clean_environment, select_node, uv_executable
from repo_dev.tasks import COMPONENTS


def read_env(path: Path):
    """Read simple dotenv assignments without executing/interpolating their values."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not match:
            continue
        name, value = match.groups()
        value = value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            end = re.search(rf"(?<!\\){quote}", value[1:])
            value = value[1 : 1 + end.start()] if end else ""
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[name] = value
    return values


def configured(value):
    if not value or not value.strip():
        return False
    if "${" in value:
        return False  # Do not claim an unexpanded variable reference is configured.
    upper = value.upper()
    return not (
        re.search(r"<[^>]+>", value)
        or "YOUR_" in upper
        or any(word in upper for word in ("CHANGEME", "REPLACE_ME", "REPLACE_WITH"))
    )


def configuration_checks(root: Path, component: str, profile: str):
    names = {
        "frontend": ("NEXT_PUBLIC_SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "APP_URL"),
        "data": ("DATABASE_URL",),
    }[component]
    # Read low precedence to high; do not print paths' contents or values.
    files = (
        [".env", ".env.development", ".env.local", ".env.development.local"]
        if component == "frontend"
        else [".env"]
    )
    values = {}
    for name in files:
        values.update(read_env(root / component / name))
    for name in names:
        if name in os.environ:
            values[name] = os.environ[name]
    required = component == "frontend" and profile == "app"
    checks = []
    for name in names:
        present = configured(values.get(name))
        checks.append(
            {
                "id": f"{component}.config.{name}",
                "status": "ok" if present else "error" if required else "warning",
                "message": f"{name}: "
                + (
                    "configured (value hidden)"
                    if present
                    else "missing/placeholder; required to run the app"
                    if required
                    else "missing/placeholder; optional for local checks/tests"
                ),
            }
        )
    return checks


def check_python(root, component):
    directory = root / component
    python = directory / ".venv/bin/python"
    if not python.exists():
        return [
            {
                "id": f"{component}.environment",
                "status": "error",
                "message": f"Missing isolated environment; run ./dev setup {component}",
            }
        ]
    checks = []
    script = "import json,sys,site; print(json.dumps({'version':list(sys.version_info[:3]), 'prefix':sys.prefix, 'base':sys.base_prefix, 'user_site':site.ENABLE_USER_SITE}))"
    probe = capture([str(python), "-I", "-c", script], env=clean_environment(directory), timeout=10)
    info = json.loads(probe.stdout) if probe.returncode == 0 else {}
    config = directory / ".venv/pyvenv.cfg"
    settings = (
        dict(line.lower().split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        if config.exists()
        else {}
    )
    settings = {key.strip(): value.strip() for key, value in settings.items()}
    isolated = (
        Path(info.get("prefix", "/")).resolve() == (directory / ".venv").resolve()
        and info.get("prefix") != info.get("base")
        and not info.get("user_site", True)
        and settings.get("include-system-site-packages") == "false"
    )
    supported = tuple(info.get("version", [])) >= (3, 12)
    checks.append(
        {
            "id": f"{component}.environment",
            "status": "ok" if isolated and supported else "error",
            "message": f"Python {'.'.join(map(str, info.get('version', [])))}; isolated={isolated}; expected {component}/.venv",
        }
    )
    lock = directory / "uv.lock"
    if not lock.exists():
        checks.append(
            {"id": f"{component}.dependencies", "status": "error", "message": "uv.lock is missing"}
        )
    else:
        proc = capture(
            [uv_executable(), "sync", "--check", "--locked", "--offline", "--no-python-downloads"],
            cwd=directory,
            env=clean_environment(directory),
            timeout=30,
        )
        checks.append(
            {
                "id": f"{component}.dependencies",
                "status": "ok" if proc.returncode == 0 else "error",
                "message": "Environment matches uv.lock"
                if proc.returncode == 0
                else f"Environment/lock is out of sync; run ./dev setup {component} (review lock changes separately)",
            }
        )
    return checks


def check_node(root, component, runtime):
    directory = root / component
    checks = [
        {
            "id": f"{component}.node",
            "status": "ok",
            "message": f"Node {'.'.join(map(str, runtime.version))} via {runtime.source}; {runtime.node}",
        }
    ]
    if not (directory / "node_modules").is_dir():
        checks.append(
            {
                "id": f"{component}.dependencies",
                "status": "error",
                "message": f"node_modules missing; run ./dev setup {component}",
            }
        )
        return checks
    package = json.loads((directory / "package.json").read_text())
    lock = json.loads((directory / "package-lock.json").read_text())
    if not isinstance(package, dict) or not isinstance(lock, dict):
        raise ValueError("package manifests must be objects")
    packages = lock.get("packages", {})
    if not isinstance(packages, dict) or not isinstance(packages.get("", {}), dict):
        raise ValueError("lockfile packages must contain object entries")
    locked = packages.get("", {})
    consistent = all(
        package.get(key, {}) == locked.get(key, {})
        for key in ("dependencies", "devDependencies", "optionalDependencies")
    )
    probe = capture(
        [str(runtime.npm), "ls", "--depth=0", "--json"],
        cwd=directory,
        env=runtime.environment(),
        timeout=30,
    )
    good = consistent and probe.returncode == 0
    checks.append(
        {
            "id": f"{component}.dependencies",
            "status": "ok" if good else "error",
            "message": "Root dependency declarations and local installation match"
            if good
            else f"Missing/invalid installed dependencies or manifest/lock drift; run ./dev setup {component}",
        }
    )
    return checks


def diagnose(root: Path, components=None, *, profile="local", ssh=False):
    components = components or list(COMPONENTS)
    if any(component not in COMPONENTS for component in components):
        raise ValueError("unknown component")
    checks = []
    checks.append(
        {
            "id": "uv",
            "status": "ok" if shutil.which("uv") else "error",
            "message": "uv is available" if shutil.which("uv") else "Install uv",
        }
    )
    for component in components:
        try:
            if component in {"tooling", "compute"}:
                checks.extend(check_python(root, component))
            else:
                checks.extend(check_node(root, component, select_node(root)))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            # Never relay raw command/config exceptions that might contain secret values.
            checks.append(
                {
                    "id": f"{component}.probe",
                    "status": "error",
                    "message": f"Could not inspect {component}; check its runtime, manifests and installation (tooling/README.md)",
                }
            )
        if component in {"frontend", "data"}:
            try:
                checks.extend(configuration_checks(root, component, profile))
            except (OSError, ValueError):
                checks.append(
                    {
                        "id": f"{component}.config",
                        "status": "error",
                        "message": "Could not read local configuration; values withheld",
                    }
                )
    if ssh:
        python = root / "compute/.venv/bin/python"
        try:
            probe = capture(
                [str(python), "-m", "kernel_compute.cli", "doctor", "--json"],
                cwd=root / "compute",
                env=clean_environment(root / "compute"),
                timeout=45,
            )
            ok = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        checks.append(
            {
                "id": "compute.ssh",
                "status": "ok" if ok else "error",
                "message": "Waterloo read-only provider check passed; no GPU allocated"
                if ok
                else "Waterloo check failed; run kcompute doctor from compute/ for details",
            }
        )
    else:
        checks.append(
            {
                "id": "compute.ssh",
                "status": "skipped",
                "message": "Remote access is opt-in: ./dev doctor --ssh",
            }
        )
    return {
        "ok": not any(check["status"] == "error" for check in checks),
        "profile": profile,
        "checks": checks,
    }
