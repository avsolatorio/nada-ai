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


@pytest.fixture(autouse=True)
def _isolated_nada_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No NADA request gate between tests (a test that builds a search-index client installs one, which also routes
    ai4data's requests through it), no pause between requests, no study-index lookup for the generation header, no
    scheduler started with the app (a scheduler test turns it on), and a scheduler lock of the test's own."""
    from nada_ai.ingest import search_index_sync
    from nada_ai.nada import gate

    monkeypatch.setenv("NADA_RECONCILE_SEARCH_INDEX_ENABLED", "false")
    monkeypatch.setenv("NADA_REQUEST_MIN_INTERVAL_MS", "0")
    monkeypatch.setenv("NADA_REQUEST_RETRY_BASE_SECONDS", "0")
    monkeypatch.setenv("NADA_RECONCILE_SEARCH_INDEX_LOCK_PATH", str(tmp_path / "scheduler.lock"))
    monkeypatch.setattr(search_index_sync, "index_generation", lambda settings: None)
    gate.uninstall()
    yield
    gate.uninstall()
