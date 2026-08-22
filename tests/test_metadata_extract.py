"""Tests for NADA metadata-extract response parsing."""

from nada_ai.filters.metadata_extract import parse_extract_response, study_to_sync_record

SAMPLE_LIST = {
    "status": "success",
    "offset": 0,
    "limit": 15,
    "total": 901,
    "has_more": True,
    "studies": [
        {
            "core_fields": {"idno": "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"},
            "filters": {
                "doctype": 1,
                "countries": [181],
                "tags": [],
            },
        },
        {
            "core_fields": {"idno": "RWA_NISR_DOC_2023_RPHC-DP_EP-GATS_SEP_EN_V1"},
            "filters": {
                "doctype": 1,
                "countries": [181],
            },
        },
    ],
}


def test_study_to_sync_record():
    rec = study_to_sync_record(SAMPLE_LIST["studies"][0])
    assert rec == {
        "idno": "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1",
        "filters": {"doctype": 1, "countries": [181], "tags": []},
    }


def test_parse_extract_list_response():
    records = parse_extract_response(SAMPLE_LIST)
    assert len(records) == 2
    assert records[0]["idno"] == "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"


def test_parse_extract_single_study_wrapper():
    data = {
        "status": "success",
        "study": SAMPLE_LIST["studies"][0],
    }
    records = parse_extract_response(data)
    assert len(records) == 1
    assert records[0]["idno"] == "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"
