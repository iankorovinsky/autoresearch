"""Run the repository's Modal scripts as subprocesses and collect their JSON."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("AUTORESEARCH_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def modal_python() -> str:
    """Interpreter that has the Modal client installed; defaults to this one."""
    return os.environ.get("AUTORESEARCH_MODAL_PYTHON") or sys.executable


@dataclass
class ScriptRun:
    ok: bool
    result: dict | None
    stdout: str
    stderr: str
    returncode: int | None
    error: str | None = None


def run_modal_script(script: str, args: list[str], *, kernel_source: str, kernel_name: str,
                     timeout_s: int) -> ScriptRun:
    """Write the kernel to a temp file, run `modal run <script> ... --output <json>`, load the JSON."""
    root = repo_root()
    script_path = root / script
    if not script_path.is_file():
        return ScriptRun(False, None, "", "", None, f"missing script {script_path}")
    with tempfile.TemporaryDirectory(prefix="autoresearch-kernel-") as tmp:
        kernel_file = Path(tmp) / kernel_name
        kernel_file.write_text(kernel_source, encoding="utf-8")
        output = Path(tmp) / "result.json"
        command = [modal_python(), "-m", "modal", "run", str(script_path),
                   "--kernel-path", str(kernel_file), "--output", str(output), *args]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            return ScriptRun(False, None, (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                             (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "", None,
                             f"timed out after {timeout_s} s")
        except OSError as exc:
            return ScriptRun(False, None, "", "", None, f"could not start Modal: {exc}")
        stdout, stderr = proc.stdout[-6000:], proc.stderr[-6000:]
        if not output.is_file():
            detail = (stderr or stdout).strip()
            return ScriptRun(False, None, stdout, stderr, proc.returncode,
                             f"Modal run exited with code {proc.returncode} and wrote no result:\n{detail[-3000:]}")
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ScriptRun(False, None, stdout, stderr, proc.returncode, f"result file was not valid JSON: {exc}")
        if isinstance(data, list):
            data = data[0] if data else {}
        return ScriptRun(True, data, stdout, stderr, proc.returncode)
