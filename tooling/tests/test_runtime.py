import os
import signal
import subprocess

import pytest

from repo_dev.runtime import clean_environment, select_node, stop


def test_child_environment_isolated_from_active_projects(tmp_path, monkeypatch):
    for name in (
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_PROJECT",
        "UV_WORKING_DIR",
        "NODE_PATH",
        "NODE_OPTIONS",
        "KCOMPUTE_LIVE",
    ):
        monkeypatch.setenv(name, "other-project")
    env = clean_environment(tmp_path / "compute")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "compute/.venv")
    assert "VIRTUAL_ENV" not in env and "PYTHONPATH" not in env and "NODE_PATH" not in env
    assert "KCOMPUTE_LIVE" not in env
    assert os.environ["VIRTUAL_ENV"] == "other-project"


def test_node_selection_uses_installed_24_without_changing_shell(tmp_path, monkeypatch):
    node_dir = tmp_path / "versions/node/v24.21.0/bin"
    node_dir.mkdir(parents=True)
    (node_dir / "node").touch()
    (node_dir / "npm").touch()
    monkeypatch.setenv("NVM_DIR", str(tmp_path))
    monkeypatch.delenv("REPO_DEV_NODE_BIN", raising=False)
    monkeypatch.setattr(
        "repo_dev.runtime.capture",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, "v24.21.0\n", ""),
    )
    before = os.environ["PATH"]
    runtime = select_node(tmp_path)
    assert runtime.node == node_dir / "node"
    assert runtime.environment()["PATH"].startswith(str(node_dir))
    assert os.environ["PATH"] == before


def test_explicit_old_node_does_not_silently_fall_back(tmp_path, monkeypatch):
    (tmp_path / "node").touch()
    (tmp_path / "npm").touch()
    monkeypatch.setenv("REPO_DEV_NODE_BIN", str(tmp_path))
    monkeypatch.setattr(
        "repo_dev.runtime.capture",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, "v16.20.2\n", ""),
    )
    with pytest.raises(ValueError, match="Node >=22.12"):
        select_node(tmp_path)


def test_cleanup_signals_group_even_after_leader_exits(monkeypatch):
    calls = []

    class ExitedLeader:
        pid = 12345

        def poll(self):
            return 0

    def killpg(pid, sig):
        calls.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr("repo_dev.runtime.os.killpg", killpg)
    stop(ExitedLeader())
    assert calls[0] == (12345, signal.SIGTERM)
