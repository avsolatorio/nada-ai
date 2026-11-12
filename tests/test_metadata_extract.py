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
            "core_fields": {"catalog_id": 11, "idno": "RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1"},
            "filters": {
                "doctype": 1,
                "countries": [181],
                "tags": [],
            },
        },
        {
            "core_fields": {"catalog_id": 12, "idno": "RWA_NISR_DOC_2023_RPHC-DP_EP-GATS_SEP_EN_V1"},
            "filters": {
                "doctype": 1,
                "countries": [181],
            },
        },
    ],
}


def test_study_to_sync_record():
    rec = study_to_sync_record(SAMPLE_LIST["studies"][0])
    assert rec == {"sid": 11, "filters": {"doctype": 1, "countries": [181], "tags": []}}


def test_a_study_without_its_id_is_not_a_sync_record():
    assert study_to_sync_record({"core_fields": {"idno": "X"}, "filters": {}}) is None


def test_parse_extract_list_response():
    records = parse_extract_response(SAMPLE_LIST)
    assert len(records) == 2
    assert [r["sid"] for r in records] == [11, 12]


def test_parse_extract_single_study_wrapper():
    data = {
        "status": "success",
        "study": SAMPLE_LIST["studies"][0],
    }
    records = parse_extract_response(data)
    assert len(records) == 1
    assert records[0]["sid"] == 11
