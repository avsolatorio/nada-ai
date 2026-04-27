"""CLI: ``python -m nada_ai.ingest.cli`` (Fire).

Examples (run from repo root with ``uv run`` so env vars apply)::

    uv run python -m nada_ai.ingest.cli create_index

    uv run python -m nada_ai.ingest.cli index --idnos=WB_123,WB_456 --metadata_type=indicator

    uv run python -m nada_ai.ingest.cli index_from_catalog --catalog_type=timeseries

OpenSearch ML (``NADA_EMBEDDING_BACKEND=opensearch_ml``)::

    uv run python -m nada_ai.ingest.cli setup_ingest_pipeline
"""

from __future__ import annotations

from nada_ai.ingest.pipeline import ensure_index, run_bulk_index
from nada_ai.search.backend.opensearch.client import build_client
from nada_ai.search.backend.opensearch.embeddings import EmbeddingService
from nada_ai.search.backend.opensearch.ml.setup import ensure_text_embedding_ingest_pipeline
from nada_ai.settings import Settings


def create_index(recreate: bool = False) -> None:
    """Create the OpenSearch index with k-NN mapping (local model dim or ``opensearch_ml_embedding_dimension``)."""
    settings = Settings()
    client = build_client(settings)
    if recreate and client.indices.exists(index=settings.index_name):
        client.indices.delete(index=settings.index_name)
    if settings.embedding_backend == "opensearch_ml":
        ensure_text_embedding_ingest_pipeline(client, settings)
        dim = int(settings.opensearch_ml_embedding_dimension or 0)
    else:
        embedding = EmbeddingService(settings)
        dim = embedding.embedding_dimension()
    ensure_index(client, settings, dim)
    try:
        client.transport.close()
    except Exception:
        pass
    print(f"Index ready: {settings.index_name} (dim={dim})")


def setup_ingest_pipeline() -> None:
    """Create or replace the ``text_embedding`` ingest pipeline (for ``embedding_backend=opensearch_ml``)."""
    settings = Settings()
    if settings.embedding_backend != "opensearch_ml":
        print("Note: embedding_backend is not opensearch_ml; pipeline may still be useful for manual indexing.")
    client = build_client(settings)
    ensure_text_embedding_ingest_pipeline(client, settings)
    try:
        client.transport.close()
    except Exception:
        pass
    print(f"Ingest pipeline ready: {settings.opensearch_ml_ingest_pipeline_name}")


def index(
    idnos: str,
    metadata_type: str = "indicator",
    force: bool = False,
    recreate_index: bool = False,
) -> None:
    """
    Bulk-index comma-separated idnos for a single metadata_type.

    Example metadata_type: indicator, document, microdata, geospatial
    """
    settings = Settings()
    ids = [x.strip() for x in idnos.split(",") if x.strip()]
    pairs = [(i, metadata_type) for i in ids]
    n, err = run_bulk_index(settings, pairs, force=force, recreate_index=recreate_index, show_progress_bar=True)
    err_part = f"{len(err)} bulk error(s)" if err else "ok"
    print(f"Indexed {n} docs; {err_part}")


def index_from_catalog(
    catalog_type: str = "timeseries",
    ps: int = 100,
    limit: int | None = None,
    force: bool = False,
    recreate_index: bool = False,
    show_progress_bar: bool = True,
    buffer_size: int = 1000,
) -> None:
    """Fetch ids from Data Compass search API and index them (via ``ai4data.discovery.catalog``).

    Set ``limit`` to only use the first N catalog rows (pagination stops early).
    """
    from ai4data.discovery.catalog import get_metadata_ids

    # Map user-facing type to catalog API
    params: dict = {"sk": "", "ps": ps, "type": catalog_type, "sort_by": "year", "sort_order": "asc"}
    if catalog_type == "indicator":
        params["type"] = "timeseries"
    elif catalog_type == "microdata":
        params["type"] = "survey"

    rows = get_metadata_ids(params, max_items=limit)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        idno = row.get("idno")
        t = row.get("type")
        if not idno or not t:
            continue
        pairs.append((idno, t))

    settings = Settings()
    n, err = run_bulk_index(
        settings,
        pairs,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
    )
    err_part = f"{len(err)} bulk error(s)" if err else "ok"
    print(f"Indexed {n} docs from catalog ({len(pairs)} ids); {err_part}")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "create_index": create_index,
            "setup_ingest_pipeline": setup_ingest_pipeline,
            "index": index,
            "index_from_catalog": index_from_catalog,
        }
    )
