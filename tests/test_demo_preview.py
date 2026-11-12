"""Tests for demo PDF page preview endpoint."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from starlette.testclient import TestClient

from nada_ai.app.demo_preview import render_pdf_page_png
from nada_ai.app.main import app


def _write_minimal_pdf(path: Path, pages: int = 2) -> None:
    doc = pymupdf.open()
    try:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {i}")
        doc.save(path)
    finally:
        doc.close()


def test_render_pdf_page_png_bytes(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf, pages=2)
    png = render_pdf_page_png(pdf, 0, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="out of range"):
        render_pdf_page_png(pdf, 9, dpi=72)


def _use_cache_dir(tmp_discovery_data_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ai4data.discovery import config as discovery_config
    from ai4data.discovery.paths import init_discovery_paths

    monkeypatch.setattr(discovery_config.discovery_data, "data_path", tmp_discovery_data_path)
    init_discovery_paths(tmp_discovery_data_path)


def _cached_pdf(idno: str, resource_id: str, filesize: int, pages: int = 3) -> Path:
    from ai4data.discovery.paths import get_document_cache_path

    path = get_document_cache_path(idno, "document", resource_id=resource_id, filesize=filesize)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_minimal_pdf(path, pages=pages)
    return path


def test_demo_page_preview_endpoint(tmp_discovery_data_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_cache_dir(tmp_discovery_data_path, monkeypatch)
    idno = "TESTDOC001"
    _cached_pdf(idno, "7", 1000, pages=3)

    with TestClient(app) as client:
        r = client.get(f"/demo/documents/{idno}/resources/7/pages/1.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

        assert client.get("/demo/documents/NO_SUCH_IDNO/resources/7/pages/0.png").status_code == 404
        # another PDF of the same document is not this one
        assert client.get(f"/demo/documents/{idno}/resources/8/pages/0.png").status_code == 404
        assert client.get(f"/demo/documents/{idno}/resources/7/pages/99.png").status_code == 400


def test_resolve_document_pdf_path_finds_the_resources_pdf(
    tmp_discovery_data_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nada_ai.app.demo_preview import resolve_document_pdf_path

    _use_cache_dir(tmp_discovery_data_path, monkeypatch)
    idno = "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"
    first = _cached_pdf(idno, "772", 3667934, pages=1)
    second = _cached_pdf(idno, "773", 1200, pages=1)

    assert resolve_document_pdf_path(idno, 772) == first
    assert resolve_document_pdf_path(idno, 773) == second
    assert resolve_document_pdf_path(idno, 77) is None
