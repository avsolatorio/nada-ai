# Variable search contract

Status: **implemented** (OpenSearch engine only, lexical only). Sibling to `docs/studies-search-contract.md`, not
part of it — see that doc's §10.

Models: `src/nada_ai/app/variables_schemas.py`. Backend: `src/nada_ai/search/backend/opensearch/variables_search.py`.

## 1. Why this is separate from the study search, and smaller

Variable search predates this project: every semantic study-search driver (`opensearch`, `qdrant_db`, `qdrant`)
already falls through to NADA's own database search for it (`Catalog_search_semantic_base.php::vsearch`/
`v_quick_search`), so there was no outage or missing feature to fix — only a lower-quality keyword match to improve.
Given that, and given the size of the ranking work the study search took (RRF fusion, semantic bounding, field-weight
tuning, an eval harness), the deliberate scope decision here was:

- **Lexical only.** No semantic or hybrid mode, no embeddings, no golden-query eval cycle. `LEXICAL_FIELDS`' weights
  (`label^10`, `name^5`, `question^3`, `categories`) are a starting point reasoned by analogy to the study search's
  own field weights, not a measured result.
- **A new, standalone index and pipeline**, not folded into the chunk/embedding machinery
  (`ingest/pipeline.py`, `ingest/opensearch_writer.py`): a variable document is a flat denormalization of one DB
  row, with no chunking and no embedding, so none of that applies. See `src/nada_ai/ingest/variables_index.py`.
- **Live delta sync now covers both ways a study's variables actually change** (added after this contract first
  shipped): a full study index (`index_ids_op`, whatever triggers it — the dashboard, a webhook, the CLI) also
  syncs that idno's variables, since a full index is study + chunks + variables together; and NADA's own
  `change_class="variables"` signal (fired by `Dataset_microdata_model::index_variable_data()` after a DDI/
  data-dictionary re-import — variables changed, nothing else about the study did) syncs *only* the variables via
  the same queue `search_index_sync.py` already polls for studies, without a full reindex. Both paths are
  best-effort: a variable-sync failure never fails the study index it rode in on. **Not covered:**
  `index_from_catalog_op` (bulk, catalog-type-at-a-time reindex) does not sync variables per idno — doing so in
  that tight loop would add a NADA round-trip per idno to a run that can cover thousands of them. Bulk operations
  still rely on running `backfill_variables` separately (see §5).

## 2. `POST /variables/search`

Request:

```json
{
  "query": "diarrhea",
  "filters": { "sids": [305], "types": ["survey"] },
  "sort": "relevance",
  "limit": 15,
  "offset": 0
}
```

- `query` — required, 1-500 characters. Unlike the study search there is no `browse` (no-query) mode: a variable
  search always matches something.
- `filters.sids` — restrict to these studies (NADA internal `surveys.id`, at most 5000). This is what a per-study
  variable search (NADA's `v_quick_search`) becomes: `sids: [that one sid]`.
- `filters.types` — NADA dataset types (`surveys.type`), any-of. This is the only filter parity gap with NADA's own
  `vsearch()`: countries, years, collections, repository and data-access-type are not implemented here yet (documented
  gap, not silently dropped — an unknown filter key is rejected, same as the study search).
- `sort` — `relevance` (default), `name` or `title`. NADA's DB search also sorts by `nation`; not implemented here.
- Only published variables (of published studies) are ever searched — implicit, not a filter, exactly like the
  study search's "only published studies."

Response:

```json
{
  "engine": "opensearch",
  "found": 41,
  "limit": 15,
  "offset": 0,
  "truncated": false,
  "hits": [
    {
      "rank": 1, "uid": 25097, "sid": 2, "idno": "EGY_2014_DHS_v01_M",
      "fid": "F1", "vid": "V4", "name": "hv002", "label": "Household number",
      "question": null, "title": "custom title goes here", "nation": "Egypt",
      "dataset_type": "survey", "year_start": 2014, "year_end": 2014, "score": 12.02
    }
  ],
  "applied": { "query": "diarrhea", "filters": {}, "sort": "relevance", "limit": 15, "offset": 0 },
  "timing_ms": { "total": 4.2, "engine": 3.0 }
}
```

`uid` is the variable's own id (`variables.uid`); `sid` is its study's NADA internal id. Every match is counted and
paged (no cap, no score cutoff — same "nothing is silently dropped" rule as the study search); `truncated` is true
when `found` exceeds `max_offset` (10,000, same limit as the study search).

## 3. `GET /info`

`capabilities.variables_search` is true whenever `search_backend=opensearch`, independently of
`studies_search`/`lexical`/`semantic`/`hybrid` (there are no modes to gate it on). An explicit request on another
engine (`qdrant`) answers `unsupported_capability` (501) — same code the study search uses for a mode it does not
implement.

## 4. Errors

Same envelope, same codes, same auth guard and rate limiter as the study search (`studies_errors.py`,
`ErrorCode` in `studies_schemas.py`) — reused directly rather than duplicated: `unknown_filter`,
`invalid_filter_value`, `offset_out_of_range`, `invalid_request`, `unsupported_capability`, `index_not_ready`,
`backend_unavailable`, `unauthorized`, `forbidden`, `rate_limited`.

## 5. Indexing

- **Backfill (the whole catalog):** `uv run python -m nada_ai.ingest.cli backfill_variables`. Pages NADA's
  `/api/admin/search-metadata-extract/variables` batch route; idempotent (`_id` is the variable's `uid`, so a
  re-run replaces rather than duplicates).
- **One study:** `uv run python -m nada_ai.ingest.cli index_survey_variables --idno=<idno>`. Deletes then
  re-indexes every variable of that study, from `/api/admin/search-metadata-extract/variables/<idno>`.
- **Templates:** `put_index_template` (the same command the study/chunk indices use) now also installs the
  variable index's composable template.
- The NADA-side extract endpoint (`variables_get()` in `Search_metadata_extract.php`) and the extract document
  builder (`build_variable_document`/`build_variables_by_survey`/`build_variable_batch` in
  `Catalog_search_metadata_extract.php`) did not exist before this work — they were a 501 stub. Field set:
  `uid, sid, fid, vid, name, label, question, categories`, denormalized with the owning study's
  `idno, title, nation, dataset_type, year_start, year_end, countries` — matching NADA's own, separate legacy
  OpenSearch variable indexer (`OpenSearch_variable_indexer.php`) rather than inventing a new shape.

## 6. What NADA does with a response

Same as the study search: hydrate nothing further from its own DB for the fields already denormalized here
(`idno`, `title`, `nation`, `year_start`, `year_end`) — `Catalog_search_semantic_studies.php`'s `vsearch()`/
`v_quick_search()` map a hit directly to the row shape its database driver already returns
(`uid, sid, fid, vid, name, labl, qstn, title, idno, nation, year_start, year_end`), so the catalog UI is unaffected
by which driver ran. `total` in that shape is **not** from this API: `vsearch()`'s total is the catalog-wide
published variable count, `v_quick_search()`'s is that one study's own count — both computed from NADA's database,
the same way the pre-existing DB driver computes them.

An empty query is never sent to this API: NADA's driver falls back to the database search directly when there is no
keyword to match (a keyword-less variable browse is a real, supported database case this lexical-only API does not
serve).

## 7. Known gaps (not silently papered over)

- No semantic/hybrid mode, no eval-measured field weights (see §1).
- No countries/years/collections/repository/data-access-type filters, and no `nation` sort (§2).
- Bulk catalog-type reindexes (`index_from_catalog_op`) don't sync variables per idno — a full catalog reindex
  still needs a separate `backfill_variables` run to pick up variable changes; only single/few-idno indexing
  (`index_ids_op`, whatever triggers it) and NADA's dedicated `change_class="variables"` signal are live-synced
  (see §1).
- `authoring_entity` (present in NADA's DB response shape) is not indexed or returned here.
