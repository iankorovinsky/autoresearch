"""Explicit task catalog. No implicit migrations, GPU jobs, or shared environments."""

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from repo_dev.runtime import clean_environment, select_node, stop, uv_executable

COMPONENTS = ("tooling", "compute", "frontend", "data")


@dataclass(frozen=True)
class Task:
    name: str
    component: str
    phase: str
    runtime: str
    args: tuple[str, ...]
    description: str


def catalog():
    tasks = []
    for component in ("tooling", "compute"):
        directories = ("src", "tests", "examples") if component == "compute" else ("src", "tests")
        tasks.extend(
            [
                Task(
                    f"{component}:setup",
                    component,
                    "setup",
                    "uv",
                    ("sync", "--locked", "--python", "3.12"),
                    "Install this component's locked dependencies",
                ),
                Task(
                    f"{component}:check",
                    component,
                    "check",
                    "uv",
                    ("run", "--locked", "--no-sync", "ruff", "check", *directories),
                    "Check Python source without rewriting files",
                ),
                Task(
                    f"{component}:test",
                    component,
                    "test",
                    "uv",
                    (
                        "run",
                        "--locked",
                        "--no-sync",
                        "pytest",
                        "-q",
                        *(("-m", "not live") if component == "compute" else ()),
                    ),
                    "Run local tests; no GPU jobs",
                ),
            ]
        )
    for component, check in (("frontend", "typecheck"), ("data", "schema:validate")):
        tasks.extend(
            [
                Task(
                    f"{component}:setup",
                    component,
                    "setup",
                    "npm",
                    ("ci",),
                    "Install this package's locked node_modules",
                ),
                Task(
                    f"{component}:check",
                    component,
                    "check",
                    "npm",
                    ("run", check),
                    "Type/schema validation; no deployment or migration",
                ),
                Task(
                    f"{component}:test",
                    component,
                    "test",
                    "npm",
                    ("test",),
                    "Run existing local tests",
                ),
            ]
        )
    tasks.extend(
        [
            Task(
                "docs:check",
                "tooling",
                "docs",
                "self",
                ("docs",),
                "Check local documentation and artifact indexes",
            ),
            Task(
                "frontend:dev",
                "frontend",
                "manual",
                "npm",
                ("run", "dev"),
                "Start the local frontend server",
            ),
            Task(
                "frontend:build",
                "frontend",
                "manual",
                "npm",
                ("run", "build"),
                "Build the frontend (may fetch remote fonts)",
            ),
        ]
    )
    return {task.name: task for task in tasks}


def select_tasks(phase, components):
    components = components or list(COMPONENTS)
    if len(set(components)) != len(components) or any(c not in COMPONENTS for c in components):
        raise ValueError("components must be unique members of: " + ", ".join(COMPONENTS))
    tasks = catalog()
    selected = [tasks[f"{component}:{phase}"] for component in components]
    if phase == "check":
        selected.append(tasks["docs:check"])
    return selected


def resolve(task: Task, root: Path):
    cwd = root / task.component
    if task.runtime == "uv":
        command = [uv_executable(), *task.args]
        env = clean_environment(cwd)
    elif task.runtime == "npm":
        runtime = select_node(root)
        command = [str(runtime.npm), *task.args]
        env = runtime.environment()
    else:
        command = [sys.executable, "-m", "repo_dev.cli", "--root", str(root), *task.args]
        env = clean_environment(root / "tooling")
    return command, cwd, env


def execute(tasks, root, *, dry_run=False, json_output=False, fail_fast=False):
    results = []
    for task in tasks:
        started = time.monotonic()
        result = {"task": task.name, "cwd": str(root / task.component)}
        try:
            command, cwd, env = resolve(task, root)
            result["command"] = command
            if dry_run:
                result.update(status="planned", exit_code=0)
            else:
                if (
                    task.runtime == "uv"
                    and task.phase != "setup"
                    and not (cwd / ".venv/bin/python").exists()
                ):
                    raise ValueError(f"missing environment; run ./dev setup {task.component}")
                if (
                    task.runtime == "npm"
                    and task.phase != "setup"
                    and not (cwd / "node_modules").is_dir()
                ):
                    raise ValueError(f"missing dependencies; run ./dev setup {task.component}")
                print(f"\n[{task.name}] {task.description}", file=sys.stderr, flush=True)
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    start_new_session=True,
                    stdout=sys.stderr if json_output else None,
                )
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    stop(process, signal.SIGINT)
                    raise
                result.update(status="passed" if code == 0 else "failed", exit_code=code)
        except (OSError, ValueError) as exc:
            result.update(status="failed", exit_code=1, message=str(exc))
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        results.append(result)
        if fail_fast and result["exit_code"]:
            break
    return {"ok": all(r["exit_code"] == 0 for r in results), "dry_run": dry_run, "tasks": results}
