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
- **Variables stay in step with their study** (added after this contract first shipped). A full study index
  (`index_sids_op`: the dashboard, a webhook, the CLI) syncs that study's variables, since a full index is study +
  chunks + variables together. So does a catalog-type run (`index_from_catalog_op`) — for its microdata rows only,
  one NADA call per microdata study (a run over thousands of documents makes none). NADA's own
  `change_class="variables"` signal (fired by `Dataset_microdata_model::index_variable_data()` after a DDI/
  data-dictionary re-import — variables changed, nothing else did) syncs *only* the variables, through the same
  queue `search_index_sync.py` polls for studies, without a full reindex. A variable sync failure never stops the
  study index it rode in on (it is reported in the result's `variables` block, `failed` per idno), but that study is
  not indexed: it is reported `failed` to NADA's `search_index_state`, its queue item is acked failed, and
  reconcile-diff counts it failed, so it is retried. A cancelled run's microdata studies count as failed the same way.
- **Removal follows the study.** Deleting a study (by idno or `sid`, including reconcile's stale deletions) deletes
  its variables with it, and recreating or dropping the index drops the variable index along with the chunk and
  study indexes (`DELETE /admin/index` used to drop only the chunk index, leaving both other indexes serving).
- **Study state is read from the study index, not stored on variables.** A variable document holds the variable
  (`uid`, `sid`, `name`, `label`, `question`, `categories`) and its study's `title` to sort by — nothing else. On every
  search, the study search's own filters (`filter_clauses` in `studies_search.py`, plus the dataset types) select the
  published studies matching the request's filters from the study index; the variable query keeps the variables of
  those studies. Publishing, unpublishing, or changing a study's collections or data access therefore touches no
  variable (`apply_study_options_*` updates the study and its chunks only). It is an **allow-list**: a study missing
  from the study index (being rebuilt, a failed write, a failed delete) hides its variables rather than exposing them,
  and a study index with no published study answers `index_not_ready`. At most 65,535 studies can match one search
  (`MAX_ALLOWED_STUDIES`; the terms aggregation and terms filter limits), past which the search is refused.

## 2. `POST /variables/search`

Request:

```json
{
  "query": "diarrhea",
  "filters": { "types": ["survey"], "countries": [4], "year_from": 2000, "year_to": 2010, "collections": ["wb"] },
  "sort": "relevance",
  "order": "asc",
  "limit": 15,
  "offset": 0
}
```

- `query` — 1-500 characters, or omitted to **browse**: filters and sort only, every published variable.
- `filters.sids` — restrict to these studies (NADA internal `surveys.id`, at most 5000). This is what a per-study
  variable search (NADA's `v_quick_search`) becomes: `sids: [that one sid]`.
- `filters.types`, `countries`, `year_from`/`year_to`, `repository`, `collections`, `form_ids` — about the variable's
  study, with exactly the study search's meaning (they are the study search's filters, applied through the study
  index): dataset types; country ids; any data collection year in range; the active repository scope and collection
  ids (primary or secondary membership); data access form ids. Any-of within a key, AND across keys. An unknown key is
  rejected (`unknown_filter`).
- `sort` — `relevance`, `name` or `title` (the study's title, then the variable name). Default: `relevance` with a
  query, `title` without one; `relevance` without a query is `invalid_request`. Every sort ends on `uid`, so rows
  never tie. Name and title sorts ignore case and accents. NADA's DB search also sorts by `nation`; not implemented.
- `order` — `asc` (default) or `desc`: the direction of a `name` or `title` sort; ignored for `relevance`.
- `limit` — 1-100. `offset` / `cursor` — see Paging. Send one or the other, not both.
- Only published variables (of published studies) are ever searched — implicit, not a filter.

### Paging

Two ways, over the same sort:

- **Offset**, for the first `variables_max_offset` rows (setting `NADA_VARIABLES_MAX_OFFSET`, default 1,000, at most
  10,000): `offset + limit` must not exceed it, else `offset_out_of_range` with `details.max_offset`. With 15 rows a
  page that is pages 1-66.
- **Cursor**, for any depth: send a response's `next_cursor` or `prev_cursor` as `cursor`, with the same query,
  filters, sort and order. It is OpenSearch `search_after` from the boundary row (a previous page is read with the
  sort reversed and flipped back), so every page costs the same however deep. A cursor is an opaque, URL-safe string:
  pass it through unchanged. It is not signed (an edited one can only move where the page starts or what its labels
  say), never expires, and needs no server state. A cursor that is not one of this search's (unreadable, edited, or
  another sort's) is `cursor_invalid`: start again from the first page. Rows may shift a little between pages if the
  index changes meanwhile, mostly for relevance.

Every response carries both cursors (null at either end), whichever way the page was reached, so a client can switch
from offset to cursor paging at `max_offset` without asking twice. A cursor page reports the `found` of the first page
of the search (carried in the cursor) and is not counted again.

### Counting

- With a query, matches are counted exactly up to `variables_count_cap` (setting `NADA_VARIABLES_COUNT_CAP`, default
  20,000, never below `variables_max_offset`); past it, `found` is the cap and `found_is_lower_bound` is true
  (show "20,000+"). Counting every match of a broad term is what a query costs most on a large index.
- A browse (no query) is a filter only, nothing is scored, and is always counted exactly.

Response:

```json
{
  "engine": "opensearch",
  "found": 4780,
  "found_is_lower_bound": false,
  "limit": 15,
  "offset": 990,
  "max_offset": 1000,
  "next_cursor": "WzEsImEiLFs4LjEyLDQ0MDE5Ml0sMTAwNSw0NzgwLGZhbHNlXQ",
  "prev_cursor": "WzEsImIiLFs4LjE5LDQ0MDExMF0sOTkwLDQ3ODAsZmFsc2Vd",
  "hits": [
    { "rank": 991, "uid": 25097, "sid": 2, "score": 8.19 }
  ],
  "applied": { "query": "diarrhea", "mode": "lexical", "filters": {}, "sort": "relevance", "order": "asc",
               "limit": 15, "paging": "cursor" },
  "timing_ms": { "total": 4.2, "engine": 3.0 }
}
```

A hit is ids only: `uid` is the variable's own id (`variables.uid`), `sid` is its study's NADA internal id, `rank`
its 1-based position in the whole list (`offset` + index + 1). `score` is null for a name or title sort. The caller
reads the variable itself from its database (section 6). The variable index stores only `uid` and `sid` in `_source`;
the other fields are indexed for search and sorting, not stored.

Each page of one search is sent to the same shard copies (`preference` = a hash of query, filters, sort and order),
so relevance scores stay consistent from page to page. A search that OpenSearch does not answer within
`search_timeout_seconds` (setting `NADA_SEARCH_TIMEOUT_SECONDS`, default 5) is `search_timeout`.

## 3. `GET /info`

`limits.variables_max_offset` and `limits.variables_count_cap` report the two paging settings above.

`capabilities.variables_search` is true whenever `search_backend=opensearch`, independently of
`studies_search`/`lexical`/`semantic`/`hybrid` (there are no modes to gate it on). An explicit request on another
engine (`qdrant`) answers `unsupported_capability` (501) — same code the study search uses for a mode it does not
implement.

## 4. Errors

Same envelope, same codes, same auth guard and rate limiter as the study search (`studies_errors.py`,
`ErrorCode` in `studies_schemas.py`) — reused directly rather than duplicated: `unknown_filter`,
`invalid_filter_value`, `offset_out_of_range`, `cursor_invalid` (422), `invalid_request`, `search_timeout` (504), `unsupported_capability`, `index_not_ready`,
`backend_unavailable`, `query_rejected`, `unauthorized`, `forbidden`, `rate_limited`.

## 5. Indexing

- **Backfill (the whole catalog):** `uv run python -m nada_ai.ingest.cli backfill_variables`. Pages NADA's
  `/api/admin/search-metadata-extract/variables` batch route; idempotent (`_id` is the variable's `uid`, so a
  re-run replaces rather than duplicates).
- **One study:** `uv run python -m nada_ai.ingest.cli index_survey_variables --sid=<sid>`. Deletes then
  re-indexes every variable of that study, from `/api/admin/search-metadata-extract/variables/<sid>?id_format=id`.
- **Variables only, as a background job:** `POST /admin/variables/sync` (role `write`, OpenSearch only; 501 on Qdrant).
  Body `{"sids": [...]}` (NADA internal ids) syncs those studies' variables; omit `sids` to walk the whole catalog. An explicitly empty
  list is a 400 rather than "everything". Nothing else is touched: no study document, no chunks, no embedding. The job
  kind is `index_variables` and appears in `/jobs` with the usual progress and cancel; only one job per distinct
  set of studies runs at a time. NADA's dashboard starts it from the Variables card
  (`POST /api/admin/semantic/variables_sync`).
- **Paging:** both NADA extract routes return variables a page at a time, keyset-paged on `uid`
  (`after_uid` in, `next_after_uid` and `has_more` out; `total` only on the first page). The page size is capped by
  `search_metadata_extract_variables_max_limit` (1000), separate from the studies cap (100). A study's sync reads its
  first page **before** deleting its existing variables, so an unreadable study keeps what it had.
- **Recreating the index** drops the variable index with the others; a microdata run (or the job above) refills it.
- **Recreating the variable index only:** `POST /admin/index/variables/recreate?confirm=true` (role `admin`,
  OpenSearch only) drops it, creates it empty with the current mapping and indexes every variable again, as one job
  (kind `recreate_variables`). For a change to the variable mapping, which an existing index never takes (a strict
  index refuses documents with a field it lacks). Shares the single-flight key of a whole-catalog sync
  (`index_variables:all`): while one runs it answers 409 and drops nothing. NADA's Semantic › Maintenance starts it
  ("Rebuild the variable index", `POST /api/admin/semantic/variables_index_recreate`).
- **Per study:** `GET /admin/variables/by-study` (role `read`, OpenSearch only) returns `{index, exists, studies: {sid: count}}` of published variables, read with a composite aggregation so any catalog size comes back whole. NADA compares it with its own per-study counts (`GET /api/admin/semantic/variables_coverage`) to list the studies to sync on the Index page.
- **Totals:** `GET /admin/variables/stats` (role `read`, OpenSearch only) returns `{index, exists, variables, studies}`
  for the published documents in the variable index — the same population NADA's own published-variable total counts.
  `exists: false` (zeros) before the first sync is a normal state, not an error. NADA's dashboard shows it beside the
  database total on the Overview page (`GET /api/admin/semantic/variables_stats`); totals only, so it cannot see a
  study whose variables changed but kept the same count.
- **Templates:** `put_index_template` (the same command the study/chunk indices use) now also installs the
  variable index's composable template.
- The NADA-side extract endpoint (`variables_get()` in `Search_metadata_extract.php`) and the extract document
  builder (`build_variable_document`/`build_variables_by_survey`/`build_variable_batch` in
  `Catalog_search_metadata_extract.php`) did not exist before this work — they were a 501 stub. Each variable is
  `{"core_fields": {uid, catalog_id, name, label, question, categories, title}}` — the variable and its study's id
  (stored as the document's `sid`) and title. Nothing else about the study is sent: it comes from the study index at
  search time.
- **Index templates:** every path that creates the study, variable or citation index (first sync, a rebuild, the
  study index rebuild) installs that index's current template first: OpenSearch merges a matching template into a
  new index, so an old template would bring back
  fields the mapping no longer has.

## 6. What NADA does with a response

Same as the study search: NADA reads the matching variables from its own database by `uid`, in the order of the
hits — `Catalog_search_semantic_studies.php`'s `vsearch()`/`v_quick_search()` load the row shape its database driver
already returns (`uid, sid, fid, vid, name, labl, qstn, title, idno, nation, year_start, year_end`), so the catalog UI
is unaffected by which driver ran. A hit whose variable (or study) is gone from the database, or whose study is not
published, is left out. `total` in that shape is **not** from this API: `vsearch()`'s total is the catalog-wide
published variable count, `v_quick_search()`'s is that one study's own count — both computed from NADA's database,
the same way the pre-existing DB driver computes them.

NADA's driver sends keyword searches and keyword-less browses here, with every sidebar filter; only a page of more
than 100 rows, a per-study listing without a keyword, or an outage is served by its database search.

## 7. Known gaps (not silently papered over)

- No semantic/hybrid mode, no eval-measured field weights (see §1).
- No `nation` sort (§2).
- A **recreate empties the variable index** and only a microdata run (or per-study index) refills it — same as
  the chunk and study indexes, whose other dataset types are also lost until their own run.
- **No variable state in NADA's `search_index_state`** (it is per study): the dashboard's coverage and diff views
  cannot show or detect a stale variable index.
- `authoring_entity` (present in NADA's DB response shape) is not indexed or returned here.
