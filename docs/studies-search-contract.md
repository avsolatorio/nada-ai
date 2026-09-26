# Standard study search contract

Status: **approved** (step 1 of `docs/opensearch-standard-search-plan.md`). No endpoint implements this yet.

Models: `src/nada_ai/app/studies_schemas.py`. Sample payloads: `tests/fixtures/studies_search/`. Contract tests:
`tests/test_studies_contract.py`.

## 1. Purpose and principles

`POST /studies/search` is the engine-agnostic search NADA uses for main search and filtering. `GET /info` tells clients
which engine is running and what it supports.

1. **Ids, not rows.** The response is a ranked list of study ids plus counts. NADA hydrates rows from its own DB, so the
   presentation is identical whichever engine is active.
2. **`sid` is the key.** `sid` is the NADA internal id (`surveys.id`). `idno` is returned only as a guard and is
   NADA's own `surveys.idno` (taken from the extract `core_fields.idno`, not from the record's schema idno, which can
   differ). NADA drops a hit whose `idno` does not match its DB row and logs it (this flags an index that no longer
   matches the DB).
3. **NADA resolves vocabulary; nada-ai matches ids.** Country names, ISO codes, regions, form codes and timeseries
   databases are turned into ids by NADA before the call. Regions become country ids.
4. **Nothing is dropped silently.** Unknown filter keys and malformed values are rejected with an error. The effective
   search is echoed in `applied`.
5. **No versioning.** The contract evolves by adding optional fields. Clients must ignore unknown response fields and
   use `GET /info` capabilities for feature detection.
6. **Only published studies** are ever searched. This is implicit and not a request filter.

## 2. Authentication and limits

- Role `read` (existing role model), key in the `X-NADA-Admin-Key` header. With no credential configured nada-ai
  answers `backend_unavailable` (503); `NADA_ADMIN_AUTH_DISABLED=true` turns auth off for local development. Same rate
  limiter as `POST /search`.
- **Open question 1** below: whether to also accept `Authorization: Bearer` (which NADA already sends today).

## 3. `GET /info`

Describes the running engine. Example: `info_opensearch.json`, `info_qdrant.json`.

| Field | Meaning |
|---|---|
| `engine` | `qdrant` \| `opensearch` \| `solr` |
| `engine_version` | Engine version string, if known |
| `capabilities` | `studies_search`, `lexical`, `semantic`, `hybrid`, `browse`, `facets`, `variables_search`, `citations_search` (booleans). They describe the standard study search only: the legacy passage search (`POST /search`) is not a capability, and a flag is true only when that mode is implemented and served. |
| `id_key` | Always `"sid"` |
| `filters` | Supported filter keys (see 5.2); empty when `studies_search` is false |
| `sort_fields` | Supported sort fields; empty when `studies_search` is false |
| `limits` | `max_limit`, `max_offset`, `semantic_window`, `max_query_length`; null when `studies_search` is false |
| `index` | `name`, `generation` (changes when the index is rebuilt or the engine switched), `studies` (distinct studies), `embedding_model`, `embedding_dim` |

Phase 1: OpenSearch reports `studies_search: true`; Qdrant reports `studies_search: false` and NADA keeps using the
legacy `POST /search` path for it.

## 4. `POST /studies/search` request

Examples: `request_minimal.json`, `request_query.json`, `request_browse_all_filters.json`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string \| null | null | Trimmed. Empty or blank means browse (filters and sort only). Max 500 characters. |
| `mode` | `auto` \| `lexical` \| `semantic` \| `hybrid` | `auto` | `auto` uses the best combination the engine supports. |
| `filters` | object | `{}` | See 5.2. |
| `sort` | `{by, order}` \| null | null | See 5.3. |
| `limit` | int 1..100 | 15 | |
| `offset` | int 0..10000 | 0 | |
| `include_facets` | bool | false | Reserved for dynamic filters. `true` gives `unsupported_capability` unless `capabilities.facets`. |
| `include_debug` | bool | false | Adds an engine-specific `debug` object to the response. |

## 5. Filters and sorting

### 5.1 Semantics

- Filters combine with **AND across keys**; values inside one key are **any-of**.
- Omitted, `null` or empty list means "no constraint".
- A non-empty list whose values match nothing returns **zero results**, not an error. This includes unknown `types`.
- Unknown keys, wrong value types and out-of-range values are rejected (`unknown_filter`, `invalid_filter_value`).
- Lists are de-duplicated. `applied.filters` echoes the normalised constraints.
- NADA must omit `repository` for the central catalog. nada-ai treats the value literally and does not special-case
  `"central"`.

### 5.2 Filter registry

The registry is defined once in code (`CANONICAL_FILTERS`) and advertised by `GET /info`. The right-hand columns show the
proposed **step 3 index field** (flat fields; names follow the NADA extract keys) and the NADA DB source.

| Request key | Type | Meaning | Index field (step 3) | NADA source |
|---|---|---|---|---|
| `types` | string list | NADA dataset types (`survey`, `timeseries`, `document`, `geospatial`, `table`, `image`, `video`, `script`, `timeseriesdb`) | `dataset_type` | `surveys.type` |
| `countries` | int list | Country ids | `countries` | `survey_countries.cid` |
| `year_from`, `year_to` | int 1..9999 | Data collection year range, inclusive. Only `year_from`: at or after. Only `year_to`: at or before. `year_from > year_to` is rejected. | `years` (range query) | `survey_years.data_coll_year` |
| `repository` | string | Active repository scope; matches primary or secondary membership | `repositories` | `surveys.repositoryid` plus `survey_repos` |
| `collections` | string list | Collection facet (repository ids), ANDed with `repository` | `repositories` | same |
| `form_ids` | int list | Data access form ids | `formid` | `surveys.formid` |
| `data_class_ids` | int list | Data classification ids | `data_class_id` | `surveys.data_class_id` |
| `tags` | string list | Study tags | `tags` | `survey_tags.tag` |
| `facets` | `{name: [term ids]}` | User-defined facets; any-of within a name, AND across names | `fq_<name>` | `survey_facets.term_id` |
| `sids` | int list | Restrict to these internal ids (used for timeseries database selection) | `sid` | `surveys.id` |
| `created_from`, `created_to` | int (unix seconds) | Created range, inclusive | `created` | `surveys.created` |

Value limits: `types` 20, `countries` 300, `collections` 100, `form_ids` 50, `data_class_ids` 50, `tags` 50,
`sids` 5000 entries; facet names match `[A-Za-z0-9_-]{1,64}`.

Derived data is deliberately not indexed. `regions` is not a filter: NADA expands a region to its country ids.

### 5.3 Sort

`sort.by` is one of `relevance`, `title`, `nation`, `year`, `popularity`, `created`, `changed`; `order` is `asc` or
`desc`.

- The effective sort is echoed in `applied.sort`. `relevance` without a query becomes the default browse sort
  (`title asc`) and adds a `sort_adjusted` warning.
- Default when omitted: `relevance desc` with a query, `title asc` without.
- With a query, non-relevance sorts apply to the selected result set (6.1), after selection.
- **Deterministic order**: after the chosen key, ties break by `year` descending, `title` ascending, then `sid`
  ascending. The same request against an unchanged index returns the same pages.

## 6. Response

Examples: `response_hybrid.json`, `response_browse.json`, `response_types_filter.json`, `response_empty.json`,
`response_degraded.json`, `response_truncated.json`.

| Field | Meaning |
|---|---|
| `engine` | Engine that answered |
| `found` | Studies in the pageable result set (6.1) |
| `limit`, `offset` | Echo of the paging window |
| `truncated` | True when `found` exceeds `limits.max_offset`: only that many results can be paged |
| `search_counts_by_type` | Distinct studies per NADA dataset type in the result set, **ignoring the `types` filter** (tab counts) |
| `hits[]` | `sid`, `idno`, `rank` (1-based across the whole set), `score`, `matched_by`, optional `passages` |
| `applied` | Effective `query`, `mode`, `filters`, `sort`, `limit`, `offset` |
| `warnings[]` | `{code, message}`; codes `semantic_unavailable`, `sort_adjusted` |
| `timing_ms` | Optional timings |
| `debug` | Present only when requested |

`hits[].score` is the engine's score for the mode that ran (keyword score in `lexical`, vector score in `semantic`,
comparable only within one response). It is null in browse, in `hybrid` (the fused head and the keyword matches after it
are ordered by different rules, so their scores could not be compared) and when the results are ordered by another sort.
`matched_by` is `["lexical"]`, `["semantic"]` or both, or `["idno"]` alone for an exact idno match (6.1); empty
in browse. `passages` lists matching pages for document
studies (`page` is 1-based, `excerpt` is whitespace-normalised text); it replaces parsing raw engine hits.

### 6.1 What `found` means

| Mode that ran | `found` | Bound |
|---|---|---|
| `browse` (no query) | Studies matching the filters, exact | None |
| `lexical` | Every study matching the query and filters, exact | None |
| `semantic` | Studies above the relevance floor and cutoff | `semantic_window` |
| `hybrid` | Every keyword match plus the semantic studies that are not keyword matches, deduped by `sid`, exact | The semantic side only: `semantic_window` |

- **Nothing is cut.** The keyword matches are never capped: `found` counts all of them. Only the semantic side is bounded
  (`limits.semantic_window`, starting at 50), because a vector search always has nearest studies.
- **Paging depth.** `offset + limit` must be at most `limits.max_offset` (10,000) in every mode, and a larger request is
  `offset_out_of_range`. `truncated` is true when `found` exceeds that depth, so the last results cannot be paged.
- **Order in `hybrid` with a relevance sort.** The best keyword matches (`fusion_window`, starting at 50) and the semantic
  studies are fused by rank (reciprocal rank fusion, equal weights): a study found by both comes first and the two lists
  otherwise alternate. Ahead of that fused order, for a query of at least three real words (stopwords aside), any keyword
  match whose title names every one of them (ignoring case and accents) is promoted first, in keyword rank order — rank
  fusion only counts position, so a study the semantic leg also returned could otherwise outrank a study that plainly
  is the answer. The other keyword matches follow, best
  first. With any other sort, the union of the semantic studies and the keyword matches is ordered by that sort, so
  `found` is the same whatever the sort.
- There is **no score cutoff on keyword matches**: every study the match rules accept is a keyword match, and a study that
  mentions the word only in a low-weight field simply ranks after the ones with it in the title.
- Choosing a tab (`types`) never changes which studies are in the result or their order. With a `types` filter, `found`
  equals the sum of `search_counts_by_type` over those types; without one, the counts add up to `found`.
- A query matching nothing returns `found: 0`, empty hits and empty counts. It is not an error.
- A single-token query is checked against every study's own idno (case- and accent-insensitively) before any scored
  search runs, in every mode; a match is the whole result (`found` = the number of studies with that idno, normally
  one). Nothing a scored search does is as precise as an exact idno match.
- The relevance floor and cutoff of the semantic side, and the two windows, are server settings, not request parameters.

### 6.2 Mode handling

- `auto` degrades gracefully: if the semantic leg is unavailable it returns lexical results with a
  `semantic_unavailable` warning, and `applied.mode` shows what ran.
- An explicit `hybrid` or `semantic` request that cannot be served returns `embedding_unavailable` (503) or
  `unsupported_capability` (501); it never degrades silently.

## 7. Errors

All errors on these routes use one envelope and never the framework default:

```json
{ "error": { "code": "unknown_filter", "message": "...", "details": { } } }
```

| Code | HTTP | When |
|---|---|---|
| `invalid_request` | 422 | Malformed body, wrong types, `limit` or query length out of range |
| `unknown_filter` | 422 | A filter key is not in the registry (`details.filters`, `details.supported`) |
| `invalid_filter_value` | 422 | A filter value is malformed or out of range (`details.filter`) |
| `offset_out_of_range` | 422 | `offset + limit` is beyond what the mode can page |
| `unsupported_capability` | 501 | The active engine cannot do this (for example `studies_search` on Qdrant) |
| `index_not_ready` | 503 | Index missing, empty or being rebuilt |
| `backend_unavailable` | 503 | The engine cannot be reached, or access cannot be checked (no credential configured, key store unreadable) |
| `embedding_unavailable` | 503 | The embedding model failed and the request required semantic search |
| `query_rejected` | 400 | The engine is up but refused the query nada-ai built (a nada-ai bug, not an outage; `details.engine_error`) |
| `unauthorized` | 401 | Missing or invalid key |
| `forbidden` | 403 | Key role too low |
| `rate_limited` | 429 | Rate limit exceeded |

## 8. What NADA does with a response

1. Hydrate `WHERE id IN (...) AND published = 1`, preserving `hits` order, without re-applying filters.
2. Verify each returned `idno` against the DB row; drop and log any mismatch or missing id. That count should be zero
   after a clean reindex.
3. Use `found` for pagination and `search_counts_by_type` for tab counts. Any other row extras (citation counts,
   variable-match badges) stay NADA-side.
4. Ignore response fields it does not know.

## 9. Testable invariants

Encoded in `StudySearchResponse.invariant_violations()` and reused by the implementation tests:

- Hit ranks are contiguous from `offset + 1`; `len(hits) <= limit`; `offset + len(hits) <= found`; no duplicate `sid`.
- Counts add up to `found` without a `types` filter; with one, `found` equals the sum over the selected types.
- `truncated` is true exactly when `found` exceeds `max_offset`.
- Browse: no score or `matched_by`. Relevance: every hit has `matched_by`.
- A nonsense query returns `found: 0` (acceptance test in step 7).

## 10. Out of scope for this contract

Dynamic facets (`include_facets` reserved), a reranker, and the admin route clean-up (including moving `{idno}` admin paths
to `{sid}`). Citations search is a separate contract: `citations-search-contract.md`.

**Variables search is now implemented, as its own, separate contract** — `POST /variables/search`, lexical only, no
modes and no relationship to the fusion/RRF machinery here — see `docs/variables-search-contract.md`. It is a
sibling contract, not part of this one: it does not add a study search mode, and `variables_search` in `GET /info`
(§3) describes it independently of `studies_search`/`lexical`/`semantic`/`hybrid`.

## 11. Decisions and open questions

| # | Topic | Status |
|---|---|---|
| 1 | Auth header (`Authorization: Bearer` as well as `X-NADA-Admin-Key`; `POST /search` currently has no authentication) | **Later** (reminder) |
| 2 | Query syntax: plain text plus quoted phrases; `+must -exclude` and `field:value`. When a query uses quotes or `+`/`-`, decide whether it still runs semantic/hybrid or switches to lexical | **Later** (reminder) |
| 3 | Cap of 100 for broad keyword queries | **Removed**: keyword matches are all returned and paged; only the semantic side is bounded (50). `truncated` now means `found` exceeds the paging depth |
| 4 | Index field names for step 3: follow the NADA extract keys (`countries`, `formid`, `repositories`, `fq_<name>`, ...) | **Agreed** |
| 5 | Tab counts field name `search_counts_by_type` | **Agreed** |
| 6 | Deep paging limit: `offset + limit <= 10,000`, for browse and for queries | **Agreed** (unchanged) |
| 7 | Unknown `types` return zero results, not an error | **Keep as is for now**, decide later |
| 8 | `include_debug`: admin role only | **Agreed** (default proposed) |
