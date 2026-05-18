"""Demo-only PDF page previews from the discovery document cache."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ai4data.discovery.paths import get_document_cache_path


def resolve_document_pdf_path(idno: str) -> Path:
    """Return cached PDF path for a document idno (does not check existence)."""
    return get_document_cache_path(idno.strip(), "document")


def render_pdf_page_png(pdf_path: Path, page_index: int, *, dpi: int = 120) -> bytes:
    """Render a single PDF page to PNG bytes (0-based ``page_index``)."""
    if page_index < 0:
        raise ValueError("page index must be non-negative")
    if dpi < 36 or dpi > 300:
        raise ValueError("dpi must be between 36 and 300")

    doc = pymupdf.open(pdf_path)
    try:
        if page_index >= doc.page_count:
            raise ValueError(f"page {page_index} out of range (document has {doc.page_count} page(s))")
        page = doc[page_index]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
