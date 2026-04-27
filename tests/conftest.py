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
