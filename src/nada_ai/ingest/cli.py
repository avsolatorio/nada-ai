"""CLI: ``python -m nada_ai.ingest.cli`` (Fire).

Examples (run from repo root with ``uv run`` so env vars apply)::

    uv run python -m nada_ai.ingest.cli create_index

    uv run python -m nada_ai.ingest.cli put_index_template

    uv run python -m nada_ai.ingest.cli index --idnos=WB_123,WB_456 --metadata_type=indicator

    uv run python -m nada_ai.ingest.cli index_from_catalog --catalog_type=timeseries

OpenSearch ML (``NADA_EMBEDDING_BACKEND=opensearch_ml``)::

    uv run python -m nada_ai.ingest.cli setup_ingest_pipeline
"""

from __future__ import annotations

from nada_ai.ingest.service import (
    create_index_op,
    index_from_catalog_op,
    index_ids_op,
    put_index_template_op,
    setup_ingest_pipeline_op,
)
from nada_ai.settings import Settings


def put_index_template() -> None:
    """Install composable index template (+ optional cluster ``action.auto_create_index`` from env)."""
    settings = Settings()
    res = put_index_template_op(settings)
    print(res)


def create_index(recreate: bool = False) -> None:
    """Create the OpenSearch index with k-NN mapping (local model dim or ``opensearch_ml_embedding_dimension``)."""
    settings = Settings()
    res = create_index_op(settings, recreate=recreate)
    print(f"Index ready: {res['index']} (dim={res['dim']})")


def setup_ingest_pipeline() -> None:
    """Create or replace the ``text_embedding`` ingest pipeline (for ``embedding_backend=opensearch_ml``)."""
    settings = Settings()
    if settings.embedding_backend != "opensearch_ml":
        print("Note: embedding_backend is not opensearch_ml; pipeline may still be useful for manual indexing.")
    res = setup_ingest_pipeline_op(settings)
    print(f"Ingest pipeline ready: {res['pipeline']}")


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
    res = index_ids_op(
        settings,
        idnos=ids,
        metadata_type=metadata_type,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=True,
    )
    err_part = f"{len(res['errors'])} bulk error(s)" if res["errors"] else "ok"
    print(f"Indexed {res['indexed']} docs; {err_part}")


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
    settings = Settings()
    res = index_from_catalog_op(
        settings,
        catalog_type=catalog_type,
        ps=ps,
        limit=limit,
        force=force,
        recreate_index=recreate_index,
        show_progress_bar=show_progress_bar,
        buffer_size=buffer_size,
    )
    err_part = f"{len(res['errors'])} bulk error(s)" if res["errors"] else "ok"
    print(f"Indexed {res['indexed']} docs from catalog ({res['rows']} ids); {err_part}")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "create_index": create_index,
            "put_index_template": put_index_template,
            "setup_ingest_pipeline": setup_ingest_pipeline,
            "index": index,
            "index_from_catalog": index_from_catalog,
        }
    )
