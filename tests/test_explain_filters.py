"""Tests for :func:`nada_ai.search.explain_filters.compute_filter_match`."""

from nada_ai.search.explain_filters import compute_filter_match


def test_filter_match_all_fields():
    sample = {
        "type": "indicator",
        "idno": "WB.1",
        "source": "WDI",
        "geographies": ["USA", "CAN"],
        "periodicity": "annual",
        "document_type": "x",
        "authors": ["a1"],
        "year_start": 2005,
    }
    filters = {
        "type": "indicator",
        "source": "WDI",
        "geographies": ["USA"],
        "periodicity": "annual",
        "document_type": "x",
        "authors": ["a1"],
        "year_start": 2000,
        "year_end": 2010,
    }
    out = compute_filter_match(sample, filters)
    assert out["all_matched"] is True


def test_filter_match_year_fail():
    sample = {"year_start": 1990}
    out = compute_filter_match(sample, {"year_start": 2000})
    assert out["all_matched"] is False
