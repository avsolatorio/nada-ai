"""``variables.py``: the variable document built from NADA's extract."""

from __future__ import annotations

import pytest

from nada_ai.search.backend.opensearch.variables import variable_bulk_action, variable_to_source


def test_variable_to_source_has_the_variable_and_its_study_title_only() -> None:
    """Everything else about the study (published, type, countries, years, idno, nation) is read from the study index
    at search time, so a change to the study never touches its variables."""
    source = variable_to_source(
        142785,
        {
            "uid": 142785,
            "catalog_id": 305,
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
    )
    assert source == {
        "uid": 142785,
        "sid": 305,
        "name": "h12d_6",
        "label": "diarrhea: public mobile clinic",
        "question": "question text",
        "categories": "1=yes 2=no",
        "title": "Agricultural census 2010",
    }


def test_the_document_fits_the_mapping() -> None:
    """The variable index is strict: a field the mapping lacks would refuse the whole document."""
    from nada_ai.search.backend.opensearch.mapping import variables_index_body

    properties = variables_index_body()["mappings"]["properties"]
    source = variable_to_source(
        1, {"catalog_id": 2, "name": "a", "label": "b", "question": "c", "categories": "d", "title": "e"}
    )
    assert set(source) <= set(properties)


def test_a_missing_catalog_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="catalog_id"):
        variable_to_source(1, {"name": "hhid"})


def test_empty_text_fields_are_omitted_not_stored_blank() -> None:
    source = variable_to_source(1, {"uid": 1, "catalog_id": 2, "name": "hhid", "question": "", "categories": None})
    assert "question" not in source
    assert "categories" not in source


def test_bulk_action_indexes_by_uid() -> None:
    action = variable_bulk_action("nada-ai-variables", 142785, {"uid": 142785, "catalog_id": 305, "name": "hhid"})
    assert action["_op_type"] == "index"
    assert action["_index"] == "nada-ai-variables"
    assert action["_id"] == "142785"
    assert action["_source"]["uid"] == 142785
