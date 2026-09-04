"""Live tests against NADA's own admin API — search-index queue and
search-metadata-extract. Unlike the other tests/integration/*.py files,
these need NO Docker/OpenSearch/Qdrant at all — only network reachability
to the configured NADA instance and an admin-capable credential. They exist
to catch exactly the class of bug found earlier in this codebase's history:
the catalog-admin OpenAPI spec documented dataset_type at the study's top
level, but the live API actually nests it under study["filters"] — a
mismatch no amount of mocked-response unit testing could have caught.

Run::

    export NADA_INTEGRATION_NADA_API=1
    export AI4DATA_METADATA_CATALOG_URL=https://your-nada-instance/index.php
    export AI4DATA_METADATA_CATALOG_EXTRACT_PATH=api/admin/search-metadata-extract
    export AI4DATA_METADATA_CATALOG_X_API_KEY=...   # admin-capable credential
    uv run pytest tests/integration/test_search_index_reconcile_live.py -m integration -v

(NADA_METADATA_EXTRACT_BASE_URL / NADA_SEARCH_INDEX_BASE_URL are optional —
only needed if your extract/search-index endpoints live at a different
host/path than the derivation from AI4DATA_METADATA_CATALOG_URL produces.)
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

_SKIP_REASON = (
    "Set NADA_INTEGRATION_NADA_API=1, AI4DATA_METADATA_CATALOG_URL, "
    "AI4DATA_METADATA_CATALOG_EXTRACT_PATH, and an admin "
    "AI4DATA_METADATA_CATALOG_X_API_KEY for the target NADA instance."
)


def _enabled() -> bool:
    return os.environ.get("NADA_INTEGRATION_NADA_API", "").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not _enabled(), reason=_SKIP_REASON)
def test_get_status_reaches_live_search_index_api():
    from nada_ai.ingest.search_index_sync import get_status
    from nada_ai.settings import Settings

    status = get_status(Settings())
    assert status.status == "success"
    assert isinstance(status.tracking_enabled, bool)
    assert "pending" in status.queue
    assert "indexed" in status.state


@pytest.mark.skipif(not _enabled(), reason=_SKIP_REASON)
def test_list_queue_reaches_live_search_index_api():
    from nada_ai.ingest.search_index_sync import list_queue
    from nada_ai.settings import Settings

    items = list_queue(Settings(), status="pending", object_type="survey", limit=5)
    assert isinstance(items, list)
    for item in items:
        assert item.object_type == "survey"
        assert item.status == "pending"


@pytest.mark.skipif(not _enabled(), reason=_SKIP_REASON)
def test_lookup_metadata_type_resolves_a_real_idno():
    """Regression test for the dataset_type-lives-under-filters bug: this
    calls the real API and the real parsing path, not a mocked response —
    it would have caught that mismatch directly."""
    from nada_ai.ingest.search_index_sync import lookup_metadata_type
    from nada_ai.settings import Settings

    idno = os.environ.get("NADA_INTEGRATION_TEST_IDNO", "WB_LSMS_001")
    result = lookup_metadata_type(Settings(), idno)
    assert result in {"indicator", "document", "geospatial", "microdata", None}
    # A None result here (for this specific known-document idno) would mean
    # either the idno no longer exists on the target instance, or the
    # dataset_type parsing broke again — either way, worth a human look, not
    # a silent skip. Only assert the exact value for the default idno so
    # callers pointing at a different instance via NADA_INTEGRATION_TEST_IDNO
    # aren't forced to match IHSN's specific catalog contents.
    if "NADA_INTEGRATION_TEST_IDNO" not in os.environ:
        assert result == "document"
