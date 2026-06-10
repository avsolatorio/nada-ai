"""Tests for Qdrant dynamic filter query translation."""

from qdrant_client.http import models as qm

from nada_ai.search.backend.qdrant.filters import filters_to_qdrant_filter
from nada_ai.search.dynamic_filters import dynamic_facet_qdrant_key, dynamic_filters_to_qdrant_conditions


def test_dynamic_filters_use_filter_facets_paths():
    conds = dynamic_filters_to_qdrant_conditions({"countries": [181], "repositoryid": "central"})
    assert len(conds) == 2
    assert all(isinstance(c, qm.FieldCondition) for c in conds)
    keys = {c.key for c in conds}
    assert keys == {
        dynamic_facet_qdrant_key("countries"),
        dynamic_facet_qdrant_key("repositoryid"),
    }


def test_filters_to_qdrant_includes_dynamic():
    flt = filters_to_qdrant_filter({"type": "document", "countries": ["181"]})
    assert flt is not None
    assert len(flt.must) == 2
    kinds = {type(c).__name__ for c in flt.must}
    assert kinds == {"FieldCondition"}
