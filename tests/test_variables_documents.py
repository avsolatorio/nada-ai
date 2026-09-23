"""``variables.py``: the variable document built from a metadata-extract record."""

from __future__ import annotations

import pytest

from nada_ai.search.backend.opensearch.variables import variable_bulk_action, variable_to_source


def test_variable_to_source_has_the_core_fields() -> None:
    source = variable_to_source(
        142785,
        {
            "uid": 142785,
            "sid": 305,
            "fid": "F1234",
            "vid": "V1989",
            "name": "h12d_6",
            "label": "diarrhea: public mobile clinic",
            "question": "question text",
            "categories": "1=yes 2=no",
            "idno": "PSE-PCBS-AGC-2010-V1.0",
            "title": "Agricultural census 2010",
            "nation": "Palestine",
            "dataset_type": "survey",
        },
        {"published": 1, "year_start": 2010, "year_end": 2010, "countries": [175]},
    )
    assert source == {
        "uid": 142785,
        "sid": 305,
        "fid": "F1234",
        "vid": "V1989",
        "name": "h12d_6",
        "label": "diarrhea: public mobile clinic",
        "question": "question text",
        "categories": "1=yes 2=no",
        "idno": "PSE-PCBS-AGC-2010-V1.0",
        "title": "Agricultural census 2010",
        "nation": "Palestine",
        "dataset_type": "survey",
        "published": 1,
        "year_start": 2010,
        "year_end": 2010,
        "countries": [175],
    }


def test_a_missing_sid_is_rejected() -> None:
    with pytest.raises(ValueError, match="sid"):
        variable_to_source(1, {"name": "hhid"}, {})


def test_empty_text_fields_are_omitted_not_stored_blank() -> None:
    source = variable_to_source(1, {"uid": 1, "sid": 2, "name": "hhid", "question": "", "categories": None}, {})
    assert "question" not in source
    assert "categories" not in source
    assert source["published"] == 0


def test_countries_drops_values_that_are_not_integers() -> None:
    source = variable_to_source(1, {"uid": 1, "sid": 2, "name": "hhid"}, {"countries": [1, "x", None, 2]})
    assert source["countries"] == [1, 2]


def test_no_countries_key_when_the_list_is_empty() -> None:
    source = variable_to_source(1, {"uid": 1, "sid": 2, "name": "hhid"}, {"countries": []})
    assert "countries" not in source


def test_bulk_action_indexes_by_uid() -> None:
    action = variable_bulk_action("nada-ai-variables", 142785, {"uid": 142785, "sid": 305, "name": "hhid"}, {})
    assert action["_op_type"] == "index"
    assert action["_index"] == "nada-ai-variables"
    assert action["_id"] == "142785"
    assert action["_source"]["uid"] == 142785
