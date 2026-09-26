"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_discovery_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point discovery caches at an isolated directory."""
    data = tmp_path / "nada-discovery"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AI4DATA_DISCOVERY_DATA_PATH", str(data))
    return data


@pytest.fixture(autouse=True)
def _isolated_admin_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no admin credential and an empty key store of its own: auth is on and nothing is
    configured, whatever the developer's shell or ``config/api_keys.json`` holds. A test that needs open access sets
    ``NADA_ADMIN_AUTH_DISABLED=true``; one that needs a key sets ``NADA_ADMIN_API_KEY`` or issues one."""
    monkeypatch.delenv("NADA_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("NADA_ADMIN_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("NADA_API_KEYS_PATH", str(tmp_path / "api_keys.json"))
