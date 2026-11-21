import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from repo_dev.doctor import configuration_checks, configured, diagnose, read_env


def test_configuration_hides_values_and_respects_precedence(tmp_path, monkeypatch):
    directory = tmp_path / "frontend"
    directory.mkdir()
    secret = "SUPER_SECRET_PUBLISHABLE_VALUE"
    (directory / ".env.local").write_text(
        f'NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY="{secret}"\nNEXT_PUBLIC_SUPABASE_URL=<url>\nAPP_URL=http://127.0.0.1:3000\n'
    )
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_URL", raising=False)
    monkeypatch.delenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.delenv("APP_URL", raising=False)
    checks = configuration_checks(tmp_path, "frontend", "app")
    assert secret not in json.dumps(checks)
    assert any(item["status"] == "error" for item in checks)
    monkeypatch.setenv("NEXT_PUBLIC_SUPABASE_URL", "https://another-secret.example")
    checks = configuration_checks(tmp_path, "frontend", "app")
    assert all(item["status"] == "ok" for item in checks)
    assert "another-secret" not in json.dumps(checks)


def test_database_configuration_optional_for_local_checks(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    checks = configuration_checks(tmp_path, "data", "local")
    assert checks[0]["status"] == "warning"
    assert "optional" in checks[0]["message"]


def test_dotenv_never_executes_or_expands(tmp_path):
    path = tmp_path / ".env"
    path.write_text("A=\"$(touch should-never-exist)\"\nB=abc # comment\nC='quoted'\n")
    result = read_env(path)
    assert result["A"] == "$(touch should-never-exist)"
    assert result["B"] == "abc" and result["C"] == "quoted"
    assert not (tmp_path / "should-never-exist").exists()
    assert not configured("${MISSING_VALUE}")


def test_no_remote_commands_unless_requested_and_no_raw_failures(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("repo_dev.doctor.check_python", lambda *a: [])

    def probe(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "SECRET_STDOUT", "SECRET_STDERR")

    monkeypatch.setattr("repo_dev.doctor.capture", probe)
    diagnose(tmp_path, ["tooling"])
    assert calls == []
    result = diagnose(tmp_path, ["tooling"], ssh=True)
    assert calls[0][-2:] == ["doctor", "--json"]
    assert not result["ok"] and "SECRET" not in json.dumps(result)


def test_doctor_reports_invalid_manifest_structure_without_crashing(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    (frontend / "node_modules").mkdir(parents=True)
    (frontend / "package.json").write_text("[]")
    (frontend / "package-lock.json").write_text("{}")
    runtime = SimpleNamespace(version=(24, 0, 0), source="test", node=Path("/node"))
    monkeypatch.setattr("repo_dev.doctor.select_node", lambda *_: runtime)
    result = diagnose(tmp_path, ["frontend"])
    assert not result["ok"]
    assert any(
        item["id"] == "frontend.probe" and item["status"] == "error" for item in result["checks"]
    )
