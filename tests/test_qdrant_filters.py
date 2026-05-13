"""Unit tests for Qdrant filter translation."""

from qdrant_client.http import models as qm

from nada_ai.search.backend.qdrant.filters import filters_to_qdrant_filter


def _must_keys(flt: qm.Filter) -> list[str]:
    assert flt.must
    keys = []
    for c in flt.must:
        assert isinstance(c, qm.FieldCondition)
        keys.append(c.key)
    return keys


def test_filters_to_qdrant_basic():
    flt = filters_to_qdrant_filter(
        {
            "type": "indicator",
            "idno": "WB.1",
            "geographies": ["USA", "CAN"],
            "source": ["WDI", "GEM"],
            "periodicity": "annual",
            "document_type": "doc",
            "authors": ["a", "b"],
            "year_start": 2000,
            "year_end": 2010,
        }
    )
    assert flt is not None
    keys = set(_must_keys(flt))
    assert keys == {
        "metadata.type",
        "metadata.idno",
        "metadata.geographies",
        "metadata.source",
        "metadata.periodicity",
        "metadata.document_type",
        "metadata.authors",
        "metadata.year_start",
    }


def test_filters_empty():
    assert filters_to_qdrant_filter({}) is None
    assert filters_to_qdrant_filter(None) is None
