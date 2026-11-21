import os
import sys

import pytest

from repo_dev.tasks import Task, catalog, execute, resolve, select_tasks


def test_default_catalog_excludes_live_gpu_and_database_mutations():
    tests = select_tasks("test", [])
    compute = next(task for task in tests if task.component == "compute")
    assert compute.args[-2:] == ("-m", "not live")
    for phase in ("setup", "test", "check"):
        assert not any(
            "migrate" in part or part in {"build", "dev", "kcompute"}
            for task in select_tasks(phase, [])
            for part in task.args
        )
    assert select_tasks("check", ["compute"])[-1].name == "docs:check"


def test_python_commands_target_component_environment(tmp_path):
    for name in ("tooling:check", "compute:test"):
        task = catalog()[name]
        args, cwd, env = resolve(task, tmp_path)
        assert cwd == tmp_path / task.component
        assert env["UV_PROJECT_ENVIRONMENT"] == str(cwd / ".venv")
        assert "--no-sync" in args and "--locked" in args


def test_dry_run_does_not_spawn_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "repo_dev.tasks.subprocess.Popen", lambda *a, **kw: pytest.fail("dry run spawned task")
    )
    result = execute(select_tasks("setup", ["tooling", "compute"]), tmp_path, dry_run=True)
    assert result["ok"] and all(item["status"] == "planned" for item in result["tasks"])
    assert not (tmp_path / "compute").exists()


def test_failure_is_aggregated_and_child_output_not_in_json_stream(tmp_path, monkeypatch, capfd):
    tasks = [
        Task("first", "tooling", "test", "self", (), "first"),
        Task("second", "tooling", "test", "self", (), "second"),
    ]

    def command(task, root):
        return (
            [
                sys.executable,
                "-c",
                f"print('CHILD_LOG'); raise SystemExit({2 if task.name == 'first' else 0})",
            ],
            tmp_path,
            os.environ.copy(),
        )

    monkeypatch.setattr("repo_dev.tasks.resolve", command)
    result = execute(tasks, tmp_path, json_output=True)
    assert not result["ok"] and len(result["tasks"]) == 2
    assert result["tasks"][0]["exit_code"] == 2
    captured = capfd.readouterr()
    assert captured.out == "" and "CHILD_LOG" in captured.err
    assert len(execute(tasks, tmp_path, json_output=True, fail_fast=True)["tasks"]) == 1


def test_duplicate_or_unknown_components_rejected():
    for components in (["compute", "compute"], ["typo"]):
        with pytest.raises(ValueError):
            select_tasks("test", components)
