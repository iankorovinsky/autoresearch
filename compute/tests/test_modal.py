import json

from kernel_compute.cli import main
from kernel_compute.providers.modal_backend import ModalProvider


def test_provider_listing_and_probe_override(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.toml"
    config.write_text('[providers.modal]\ngpu = "L4"\n')
    monkeypatch.setenv("KCOMPUTE_SSH_USER", "waterloo-user")
    assert main(["providers", "list", "--config", str(config)]) == 0
    providers = {value["id"]: value for value in json.loads(capsys.readouterr().out)}
    assert "waterloo-datasci" in providers
    modal = providers["modal"]
    assert modal["gpu"] == "L4"
    assert {"T4", "H100!", "B200+", "B300"} <= set(modal["gpu_types"])

    monkeypatch.setattr(ModalProvider, "probe", lambda self: self.snapshot())
    assert main(["probe", "--gpu", "t4", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["gpu"] == "T4"


def test_gpu_is_rejected_for_waterloo(tmp_path, capsys):
    assert main(["doctor", "--gpu", "T4", "--config", str(tmp_path / "missing.toml")]) == 1
    assert "--gpu requires --provider modal" in json.loads(capsys.readouterr().out)["error"]
