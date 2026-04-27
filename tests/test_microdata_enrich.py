from nada_ai.ingest.microdata_enrich import extract_microdata_enrichment_from_raw


def test_extract_finds_labels_in_nested_dict():
    raw = {
        "data_dictionary": {
            "variables": [
                {"labl": "Household income"},
                {"label": "Region code"},
            ]
        }
    }
    text = extract_microdata_enrichment_from_raw(raw)
    assert "Household income" in text
    assert "Region code" in text
