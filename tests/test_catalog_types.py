"""Every NADA dataset type must be wired end to end: catalog listing, stored type, reconcile mapping."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nada_ai.app.admin import _CATALOG_TYPES, _STORED_TYPE_BY_CATALOG_TYPE
from nada_ai.ingest.search_index_sync import _DATASET_TYPE_TO_METADATA_TYPE
from nada_ai.settings import Settings

#: NADA's surveys.type values (see Datasets.php / Catalog_dataset_import.php in NADA). Citations are a
#: separate search-index object and are intentionally not ingested.
NADA_DATASET_TYPES = {
    "survey",
    "timeseries",
    "timeseriesdb",
    "document",
    "geospatial",
    "table",
    "script",
    "image",
    "video",
}


def test_catalog_types_cover_every_nada_dataset_type():
    assert set(_CATALOG_TYPES) == NADA_DATASET_TYPES


def test_every_catalog_type_has_stored_type_and_reconcile_mapping():
    for catalog_type in _CATALOG_TYPES:
        assert catalog_type in _STORED_TYPE_BY_CATALOG_TYPE, catalog_type
        assert catalog_type in _DATASET_TYPE_TO_METADATA_TYPE, catalog_type


def test_stored_type_matches_the_metadata_type_ingest_uses():
    # ingest stores documents under the metadata_type the reconcile mapping resolves to; the two maps drifting
    # apart would make type-counts compare against the wrong stored type.
    for catalog_type in _CATALOG_TYPES:
        assert _STORED_TYPE_BY_CATALOG_TYPE[catalog_type] == _DATASET_TYPE_TO_METADATA_TYPE[catalog_type], catalog_type


@pytest.mark.parametrize(
    ("catalog_type", "expected_api_type"),
    [
        ("timeseries", "timeseries"),
        ("indicator", "timeseries"),
        ("survey", "survey"),
        ("microdata", "survey"),
        ("timeseriesdb", "timeseriesdb"),
        ("indicator-db", "timeseriesdb"),
        ("timeseries-db", "timeseriesdb"),
        ("table", "table"),
        ("script", "script"),
        ("image", "image"),
        ("video", "video"),
    ],
)
def test_index_from_catalog_op_queries_nada_with_its_own_type_name(
    catalog_type, expected_api_type, tmp_path, monkeypatch
):
    import nada_ai.ingest.service as service_module

    monkeypatch.setenv("NADA_INGEST_CHECKPOINT_DIR", str(tmp_path))
    seen_params: list[dict] = []

    def fake_get_metadata_ids(params, **kwargs):
        seen_params.append(dict(params))
        return []

    with (
        patch("ai4data.discovery.catalog.get_metadata_ids", fake_get_metadata_ids),
        patch("ai4data.discovery.catalog.is_extract_mode", return_value=False),
        patch.object(service_module, "run_bulk_index", return_value=(0, None)),
    ):
        service_module.index_from_catalog_op(Settings(), catalog_type=catalog_type, show_progress_bar=False)

    assert [p["type"] for p in seen_params] == [expected_api_type]


def test_pinned_ai4data_can_handle_every_metadata_type_nada_ai_maps_to():
    handler = pytest.importorskip("ai4data.discovery.metadata.handler")
    if not hasattr(handler, "TemplatedMetadata"):
        pytest.skip("pinned ai4data predates the table/script/image/video/indicator-db handlers — bump the pin")

    for metadata_type in set(_DATASET_TYPE_TO_METADATA_TYPE.values()):
        loader = handler.MetadataLoader.__new__(handler.MetadataLoader)
        loader.type, loader.metadata, loader.searchpath = metadata_type, {"type": metadata_type}, None
        assert loader.get_metadata_handler().type == metadata_type
