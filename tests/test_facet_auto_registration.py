"""Tests for facet-key auto-registration: new keys observed in NADA's filters
data get automatically promoted to facetable (and indexed, on Qdrant) without
a human editing the registry — see nada_ai.filters.sync.auto_register_new_facet_keys.

All tests use an isolated tmp_path-backed registry file so they never touch
the real repo config/dynamic_filter_facets.json.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from nada_ai.filters.facets_service import add_facet_keys, remove_facet_keys
from nada_ai.filters.sync import auto_register_new_facet_keys, sync_filters_for_idno
from nada_ai.search.dynamic_filters import load_dynamic_facet_keys, load_excluded_facet_keys
from nada_ai.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    path = tmp_path / "dynamic_filter_facets.json"
    return Settings(dynamic_filter_facets_path=str(path), **overrides)


# ---------------------------------------------------------------------------
# auto_register_new_facet_keys
# ---------------------------------------------------------------------------

def test_auto_register_promotes_new_key(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    new = auto_register_new_facet_keys(settings, ["brand_new_key", "countries"])
    assert new == ["brand_new_key"]  # "countries" is already in the default facetable set
    assert "brand_new_key" in load_dynamic_facet_keys(settings)


def test_auto_register_noop_when_all_keys_already_known(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    add_facet_keys(settings, ["already_known"])
    new = auto_register_new_facet_keys(settings, ["already_known"])
    assert new == []


def test_auto_register_skips_excluded_keys(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    add_facet_keys(settings, ["suppressed_key"])
    remove_facet_keys(settings, ["suppressed_key"])  # now on the deny-list

    new = auto_register_new_facet_keys(settings, ["suppressed_key"])

    assert new == []
    assert "suppressed_key" not in load_dynamic_facet_keys(settings)
    assert "suppressed_key" in load_excluded_facet_keys(settings)


def test_auto_register_skips_fixed_filter_keys(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    new = auto_register_new_facet_keys(settings, ["type", "idno", "source"])
    assert new == []
    assert load_dynamic_facet_keys(settings).isdisjoint({"type", "idno", "source"})


@patch("nada_ai.filters.sync.qdrant_client")
def test_auto_register_indexes_new_key_on_qdrant(mock_client_fn, tmp_path):
    client = MagicMock()
    mock_client_fn.return_value = client
    settings = _settings(tmp_path, search_backend="qdrant")

    new = auto_register_new_facet_keys(settings, ["brand_new_key"])

    assert new == ["brand_new_key"]
    client.create_payload_index.assert_called_once()
    kwargs = client.create_payload_index.call_args.kwargs
    assert kwargs["field_name"] == "metadata.filter_facets.brand_new_key"
    client.close.assert_called_once()


@patch("nada_ai.filters.sync.qdrant_client")
def test_auto_register_no_qdrant_call_when_nothing_new(mock_client_fn, tmp_path):
    client = MagicMock()
    mock_client_fn.return_value = client
    settings = _settings(tmp_path, search_backend="qdrant")

    new = auto_register_new_facet_keys(settings, ["countries"])  # already default-facetable

    assert new == []
    mock_client_fn.assert_not_called()


# ---------------------------------------------------------------------------
# remove/add round-trip on the excluded (deny) list
# ---------------------------------------------------------------------------

def test_remove_then_add_clears_exclusion(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    add_facet_keys(settings, ["some_key"])
    remove_facet_keys(settings, ["some_key"])
    assert "some_key" in load_excluded_facet_keys(settings)

    add_facet_keys(settings, ["some_key"])  # explicit re-add wins over prior exclusion
    assert "some_key" not in load_excluded_facet_keys(settings)
    assert "some_key" in load_dynamic_facet_keys(settings)


def test_unrelated_registry_write_preserves_excluded_list(tmp_path):
    """save_dynamic_facet_keys (e.g. via set_facets_config) must not silently
    wipe the excluded list when it only intends to touch facetable keys."""
    settings = _settings(tmp_path, search_backend="opensearch")
    remove_facet_keys(settings, ["doctype"])  # doctype -> excluded
    assert "doctype" in load_excluded_facet_keys(settings)

    add_facet_keys(settings, ["unrelated_key"])  # writes facetable, must preserve excluded

    assert "doctype" in load_excluded_facet_keys(settings)


# ---------------------------------------------------------------------------
# sync_filters_for_idno end-to-end: same-study update introduces a new key
# ---------------------------------------------------------------------------

@patch("nada_ai.filters.sync.qdrant_client")
def test_sync_registers_new_key_then_updates_same_idno_with_more_keys(mock_client_fn, tmp_path):
    """Simulates NADA sending updated filters for the same study across two
    syncs: first sync has one key, second sync adds a brand-new key. Both
    should register/index correctly, and the second sync's full-replace write
    should reflect the updated (superset) filter set for that idno."""
    client = MagicMock()
    mock_client_fn.return_value = client
    client.count.return_value = MagicMock(count=1)
    settings = _settings(tmp_path, search_backend="qdrant")

    res1 = sync_filters_for_idno(settings, "DOC-1", {"countries": [181]})
    assert res1["found"] is True
    assert "brand_new_topic" not in load_dynamic_facet_keys(settings)

    # NADA now returns updated filters for the same idno, including a new key
    res2 = sync_filters_for_idno(settings, "DOC-1", {"countries": [181], "brand_new_topic": ["health"]})
    assert res2["found"] is True
    assert "brand_new_topic" in load_dynamic_facet_keys(settings)

    # second write reflects the full updated set (last set_payload call)
    last_payload = client.set_payload.call_args_list[-1].kwargs["payload"]
    keys_written = {entry["key"] for entry in last_payload["filter_fields"]}
    assert keys_written == {"countries", "brand_new_topic"}


def test_json_file_persists_facetable_and_excluded_shape(tmp_path):
    settings = _settings(tmp_path, search_backend="opensearch")
    add_facet_keys(settings, ["kept_key", "removed_key"])
    remove_facet_keys(settings, ["removed_key"])

    path = tmp_path / "dynamic_filter_facets.json"
    data = json.loads(path.read_text())
    assert "kept_key" in data["facetable"]
    assert "removed_key" not in data["facetable"]
    assert "removed_key" in data["excluded"]
