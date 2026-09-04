from __future__ import annotations

import json
from pathlib import Path

import pytest

from kasa.errors import ConfigError
from kasa.vault import Vault, check_placement, clear_cache, fingerprint, resolve


@pytest.fixture(autouse=True)
def no_cached_vault() -> None:
    clear_cache()
    yield
    clear_cache()


def test_save_uses_private_file_and_directory_modes(tmp_path: Path) -> None:
    directory = tmp_path / "share" / "kasa"
    directory.mkdir(parents=True, mode=0o755)
    vault = Vault(directory / "vault.json")
    vault.set("API_KEY", "secret-value-long-enough")
    vault.save()

    assert vault.path.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    loaded = Vault.load(vault.path)
    assert loaded.get("API_KEY") == "secret-value-long-enough"
    assert loaded.entries()[0].fingerprint == fingerprint("secret-value-long-enough")
    assert "secret-value-long-enough" not in repr(loaded.entries())


@pytest.mark.parametrize("target", ["file", "directory"])
def test_load_refuses_permissions_readable_by_other_users(tmp_path: Path, target: str) -> None:
    directory = tmp_path / "vault"
    directory.mkdir(mode=0o700)
    path = directory / "vault.json"
    path.write_text(json.dumps({"version": 1, "secrets": {}}))
    path.chmod(0o600)
    (path if target == "file" else directory).chmod(0o644 if target == "file" else 0o755)

    with pytest.raises(ConfigError, match="other users"):
        Vault.load(path)


def test_environment_wins_over_vault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "vault" / "vault.json"
    vault = Vault(path)
    vault.set("API_KEY", "stored-secret-value")
    vault.save()
    monkeypatch.setenv("KASA_VAULT", str(path))
    monkeypatch.setenv("API_KEY", "exported-secret-value")

    assert resolve("API_KEY") == "exported-secret-value"
    monkeypatch.delenv("API_KEY")
    assert resolve("API_KEY") == "stored-secret-value"


def test_vault_is_refused_inside_memory_clone(tmp_path: Path) -> None:
    clone = tmp_path / "memory"
    clone.mkdir()
    with pytest.raises(ConfigError, match="inside the long-term memory repo"):
        check_placement(clone / "private" / "vault.json", clone_path=clone)


def test_entries_never_serialize_values(tmp_path: Path) -> None:
    vault = Vault(tmp_path / "vault.json")
    vault.set("NOTION", "notion-secret-value")
    assert "notion-secret-value" not in repr(vault.entries())
