"""Demo-only PDF page previews from the discovery document cache."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ai4data.discovery.paths import get_document_cache_path

# Cache root is derived once from a known-safe sentinel idno.  All resolved
# paths must be descendants of this directory.
_CACHE_ROOT: Path | None = None


def _cache_root() -> Path:
    global _CACHE_ROOT
    if _CACHE_ROOT is None:
        _CACHE_ROOT = get_document_cache_path("__sentinel__", "document").parent.parent
    return _CACHE_ROOT


def _assert_within_cache(path: Path) -> None:
    """Raise ValueError if *path* escapes the document cache directory."""
    try:
        path.resolve().relative_to(_cache_root().resolve())
    except ValueError:
        raise ValueError("path is outside the document cache directory")


def resolve_document_pdf_path(idno: str, resource_id: str | None = None) -> Path:
    """Return cached PDF path for a document idno (does not check existence)."""
    idno = idno.strip()
    if resource_id is not None:
        path = get_document_cache_path(idno, "document", resource_id=resource_id.strip())
        _assert_within_cache(path)
        return path

    legacy = get_document_cache_path(idno, "document")
    _assert_within_cache(legacy)
    if legacy.exists():
        return legacy

    doc_dir = legacy.parent
    pattern = f"document_{idno}--*.pdf"
    matches = sorted(doc_dir.glob(pattern))
    if matches:
        return matches[0]

    return legacy


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
