# OpenSearch option and standard study search — plan

Branch: `feat/opensearch` (created from `feat/additional-catalog-types`).
Status: steps 0-8 done (the hybrid-test milestone is reached and the defaults are tuned). Step 9 (docs and compose) is next.
Contract: `docs/studies-search-contract.md` (approved).

## Goal

Add OpenSearch as a second search engine for nada-ai, exposing a **standard, engine-agnostic study search**
that NADA can use for main search and filtering. NADA hydrates rows from its own DB, so the engine only has
to return ranked internal ids and totals. Qdrant keeps its current behaviour.

Long term nada-ai supports Qdrant, OpenSearch and Solr, **one active at a time**. Switching engine always
requires a full reindex (NADA tracks the index state for a single driver).

## Decisions

- Only one engine is active per nada-ai deployment (`NADA_SEARCH_BACKEND`).
- NADA built-in OpenSearch/Solr drivers are not touched for now.
- Qdrant keeps the current NADA semantic driver (Qdrant plus DB gap-filling).
- NADA gets a setting `semantic_search_engine` (`qdrant` | `opensearch`, default `qdrant`, no `auto`).
  NADA does not verify it against nada-ai.
- No API versioning in nada-ai (`/v1`, `/v2`). New features are discovered through capability flags on `GET /info`.
- Standard study search is `POST /studies/search`. Variables and citations follow later as
  `/variables/search` and `/citations/search` (501 until implemented).
- `POST /search` (with `/search/explain`) is kept as is: passage-level, engine-native shape, used by the demo,
  the MCP server and the legacy Qdrant driver.
- Rows are hydrated from the NADA DB. The internal id (`sid`, `surveys.id`) is the key everywhere; `idno` is kept
  only as a guard and for display. NADA resolves vocabulary (regions to country ids, names/ISO to ids, form codes
  to form ids); nada-ai only matches ids.
- Totals (`found`, per-type tab counts) come from OpenSearch. The study index has one document per study, so counts
  are exact studies, not chunk documents.
- Filters are stored as **flat fields** (one keyword/integer field per filter), not as nested `filter_fields`. This is
  how NADA's built-in Solr and OpenSearch drivers store them. Nested storage creates hidden documents (about 12 per
  document) and slower filter queries. Reminder: apply this in step 3 (see "Flat filter fields" below).
- Sidebar filters and facets stay DB-built in phase 1. Dynamic filters from OpenSearch come later
  (reserve `include_facets` in the request).

## Steps

| # | Step | Output | Done when |
|---|---|---|---|
| 0 | Branch and baseline | Branch `feat/opensearch`; fresh OpenSearch 3.6 on a separate port; nada-ai API on a separate port | Existing tests pass; the current OpenSearch backend ingests and searches a small sample |
| 1 | Contract first, no logic | `docs/studies-search-contract.md`, Pydantic models, sample JSON fixtures for `GET /info` and `POST /studies/search`; canonical filter keys, fail-closed rules, error codes, meaning of `found`, tab counts, `truncated` | Contract reviewed and approved |
| 2 | `sid` plumbing in ingest **(done)** | `metadata.sid` on every document (both engines), read from the NADA metadata-extract data (`core_fields.survey_uid`, carried by ai4data as `_extract_core_fields`); ingest requires extract mode and skips (and reports) a study without a `sid`; OpenSearch mapping and Qdrant payload index for `sid`; `delete_by_sid_op` / `delete_by_sids_op`. Document ids are unchanged (see findings). | Unit tests plus a live OpenSearch 3.6 test: two studies indexed, one deleted by `sid`, the others untouched |
| 3 | OpenSearch index model **(done)** | Study index (one document per study, `_id` = `sid`, NADA's idno, typed flat filters, sort keys, analyzers) plus the chunk index, written by one ingest job (`OpenSearchIngestWriter`); flat `filter_facets` replace nested `filter_fields` on both; each index stamped with a generation (`mappings._meta`), the chunk index also with the embedding model and dimension; one template per index | Live OpenSearch 3.6 test and a real-data run: 11 study documents and 19 chunk documents, Lucene document counts equal real counts (no nested inflation), templates do not leak between indexes |
| 4 | `GET /info` **(done)** | Engine, version, capabilities, supported filters and sorts, limits, index summary; Qdrant reports study search unsupported | Endpoint answers on both engines (checked live against OpenSearch 3.6 and the dev Qdrant); errors use the contract envelope |
| 5 | Browse path **(done)** | Filter-only `POST /studies/search` with all sorts, exact `found`, tab counts, contract errors | Live oracle test on real OpenSearch; on the real sample, 139 of 140 filter cases equal NADA's MySQL driver (the one difference is a NADA behavior, see findings) |
| 6 | Lexical leg **(done)** | Native boosts, fuzziness, `minimum_should_match`; a bounded, exact relevance cut set | Live tests on real OpenSearch; on the real sample gibberish returns 0, typos still match, real queries return small sets |
| 7 | Semantic leg, fusion, policy **(done)** | Vector leg collapsed to `sid`, rank fusion, floor, relative cutoff, cap, exact counts, stable pagination; graceful degradation; prune-by-`sid` on re-index | Live tests on real OpenSearch; on the real sample with the real embedding model: gibberish returns 0, real queries return bounded sets, semantic-only matches appear |
| 8 | Evaluation harness **(done)** | 67 golden queries; lexical-only versus semantic versus hybrid; report and tuned defaults | Report in `docs/opensearch-search-evaluation.md` |
| 9 | Docs and compose | OpenSearch compose profile, env docs, OpenAPI snapshot in CI | A fresh clone can run the OpenSearch setup from the docs |

Checkpoints: after step 1 (contract approval), after step 5 (browse and filters demo), after step 7 (hybrid test).

### Acceptance criteria (steps 5-7)

- A nonsense query returns `found = 0`.
- Every filter's result ids are a subset of the DB driver result for the same filter.
- Per-type tab counts add up to `found`.
- Pages are stable: a repeated page returns the same ids.
- The hydration drop count is zero after a clean reindex.
- Latency stays within an agreed budget.

### Defaults (tuned in step 8, see the evaluation report)

Each leg retrieves its top 200 candidates; the final result set is capped at 100.

## Step 2 findings

- **Document ids stay content-based** (`get_langdoc_uuid`) instead of the planned `sid`-prefixed ids. Qdrant point ids
  must be UUIDs, and changing them would duplicate points in existing collections on re-ingest. Re-ingesting an
  existing collection now backfills `metadata.sid` in place. `sid` (a stored, indexed field) gives the replace and
  delete guarantees; the step 3 study index will use `_id = sid`.
- **`metadata.idno` is not always NADA's idno.** ai4data takes the idno from the record's own schema, so for
  `PC11_A02-28-v22` (NADA idno) the stored `metadata.idno` is `PC11_A02-28`. Consequences: deleting by NADA idno
  misses those documents (a stale-document risk in the existing idno-keyed routes), and the contract's `idno` guard
  must compare against NADA's `surveys.idno`. The step 3 study index takes `idno` from the extract `core_fields.idno`,
  and chunk-level `metadata.idno` must not be used as a key or guard.
- **`sid` source.** NADA's metadata-extract API provides it (`core_fields.survey_uid`, single study and list; the
  queue items also carry `object_id`). The plain catalog JSON (`api/catalog/json/{idno}`) has no internal id, so
  indexing requires extract mode and fails fast without it. There is no fallback lookup: ai4data now carries the
  extract's `core_fields` (`_extract_core_fields`) and nada-ai reads `sid` from it. A study whose extract data has
  no valid `sid` is skipped and reported in `load_errors` (`stage: "extract"`), never indexed without one.
- **ai4data change (our fork `mah0001/ai4data`, branch `feat/opensearch`, commit `4195d14`, pushed):**
  `study_to_catalog_metadata` keeps `core_fields`, and `study_to_search_row` reads `survey_uid` (it looked for
  `core_fields.id`, which NADA never sends, so its search-row id was `None`). nada-ai's pin in `pyproject.toml` and
  `uv.lock` now points at that commit. We always use our own fork. ai4data's on-disk metadata cache from before the
  change lacks `_extract_core_fields`; re-fetch with `force`.
- **State reports by id** are deferred: reports are still keyed by idno, and `delete_by_sid_op` does not report to
  NADA. Sending `object_id` needs a NADA-side change and the `sid` map carried out of `run_bulk`.
- **Existing indexes** are rebuilt, not migrated (no backward compatibility is kept).

## Flat filter fields and the study index (step 3, done)

- **Layout.** Both indexes store one field per filter key under `filter_facets` (`metadata.filter_facets.<key>` on
  chunks, `filter_facets.<key>` on studies). Key names are NADA's extract keys. One shared mapping
  (`filter_facets_mapping`): known integer keys (`countries`, `years`, `formid`, ...), keyword keys
  (`dataset_type`, `repositories`, `tags`, ...), `fq_<facet>` typed integer and any other key a keyword, by dynamic
  template. No nested fields anywhere.
- **Study index** (`<index_name>-studies`, override with `NADA_STUDIES_INDEX_NAME`): `sid`, NADA `idno`, text fields
  (`nada_text` analyzer: lowercase and asciifolding, no stemming yet), `title_sort` / `nation_sort` normalized keywords,
  year, created/changed, popularity and variable counts, `filter_facets`. Strict mapping, so a typo in the builder fails.
  A study is written as soon as its extract data loads, even if it produced no chunk documents.
- **Writer.** One job writes chunks, then one study document per study (`_id` = `sid`, so a re-index replaces it).
  `recreate_index` drops and recreates both. Delete by `sid` or by idno covers both indexes; delete by idno finds the
  `sid` in the study index, so chunks stored under a different schema idno are removed too.
- **Templates.** One per index, matching the exact index name (the old `<index>-*` pattern would have applied the chunk
  mapping to the study index).
- **Removed.** The nested `filter_fields` mapping and queries on OpenSearch, `ensure_opensearch_filter_fields_mapping`,
  and the `NADA_SYNC_FILTERS_DURING_INGEST` switch (filters are always taken from the extract; the study index cannot
  work without them). Qdrant still stores the nested `filter_fields` rows next to the flat map.
- **Filter sync and admin helpers** (`sync_filters_for_idno`, `get_filter_fields_for_idno`,
  `ensure_filter_indexes_op`) now read and write the flat map, and sync also updates the study document.

**Known gap (not in scope of step 3): replacing a study.** Chunk document ids are content hashes, so re-indexing a
study whose text changed leaves the old chunks behind until the study is deleted. Counts are unaffected (they come
from the study index) but the semantic leg could surface a stale passage. A clean fix is a prune-by-`sid` step after
each study is written; decide when the semantic leg is built (step 7).

## Step 4 findings

- **Capabilities are computed, not declared.** `IMPLEMENTED_STUDY_MODES` (in `app/info.py`) lists the study-search modes
  each engine implements; `/info` derives every flag, the filter list, the sort fields and the limits from it. It is
  empty today, so `/info` reports `studies_search: false` on both engines. Steps 5-7 add `browse`, `lexical`,
  `semantic` and `hybrid` for OpenSearch together with the code that serves them. Qdrant stays empty.
- **Contract errors on these routes.** `app/studies_errors.py` renders `StudiesApiError` as the contract envelope and
  provides `studies_guard`, which turns rate limiting and authentication failures (401/403/429) into the envelope too.
  `POST /studies/search` will use the same guard and handler.
- **Index summary.** OpenSearch: the study index name, its generation, its study count, and the embedding model and
  dimension from the chunk index `_meta`; missing indexes give empty fields, not an error. Qdrant: the collection,
  the embedding model from settings and the dimension from the collection; no generation and no study count.
- **New settings:** `NADA_STUDIES_RESULT_CAP` (default 100, reported as `limits.query_result_cap`) and, from step 3,
  `NADA_STUDIES_INDEX_NAME`.
- **Contract fix.** The Qdrant fixture said `semantic: true`; the capability flags describe the standard study search
  only, so it is `false` (the contract text now says so).
- **Dev container note.** The dev Qdrant container (`nada-ai-api-qdrant-dev`) mounts the working tree
  (`_symlink/nada-ai/src`) and runs with `--reload`, so it already serves this branch's code, but its image still has
  the old ai4data pin. Searching is unaffected; indexing through it fails until the image is rebuilt with the new pin
  (indexing now requires `_extract_core_fields`).

## Step 5 findings

- **Browse on OpenSearch** (`search/backend/opensearch/studies_search.py`, route in `app/studies_search.py`): every filter
  is a flat clause on the study index. `types` is a `post_filter`, so `found` honors it while the per-type counts
  (a terms aggregation) ignore it, exactly as the contract says, with exact totals (`track_total_hits`). Sorts map to
  the index sort keys with deterministic tie-breakers (`year_start` desc, title asc, `sid` asc) and missing values last.
- **Modes are tied to executors.** `EXECUTORS` lists the modes the engine serves; `GET /info` derives its capabilities
  from it, so `browse` became advertised the moment its executor existed. A query is answered with
  `unsupported_capability` until step 6 adds a lexical executor; `include_facets` likewise.
- **Contract errors.** A validation handler maps request errors to the contract codes on this route only (unknown
  filter key, invalid filter value, offset out of range, invalid request); other routes keep the framework body.
  Missing or empty study index is `index_not_ready`, an unreachable engine `backend_unavailable`, `include_debug`
  needs the admin role, and access errors come before validation errors.
- **Verification.** (1) A live test indexes synthetic studies through the real writer and compares `browse` with a
  plain-Python oracle for 23 filter combinations, every sort in both directions (ties, accents, missing values) and
  stable paging. (2) On the real sample, 140 filter cases (each value that occurs in the data, for every filter kind
  except `created`) were compared with NADA's MySQL driver restricted to the indexed studies: 139 identical.
- **The one difference is NADA's, not the API's.** `repo=string`: NADA's `set_active_repo` silently ignores an active
  repository it does not list and returns the whole catalog (1391), while the DB has 12 published studies in that
  repository (1 in the sample, which is what the API returned; NADA's `collection=string` agrees). Silently dropping an
  invalid `repo` is the fail-open pattern the contract rejects; worth fixing on the NADA side when the new driver is
  written.

## Step 6 findings

- **The query** (`lexical_query`): `multi_match`, `most_fields`, over `idno.text^60, title^40, nation^30,
  authoring_entity^10, keywords^10, abstract, methodology, var_keywords^15` (NADA's own boosts), `minimum_should_match
  2<75%`, `fuzziness AUTO`, `prefix_length 2`. `idno` is a keyword with a `text` subfield (`nada_text`), so it can be
  matched and boosted while `terms idno` lookups still work. User text is data: a `multi_match` never interprets
  operators or quotes (contract question 2 stays open: phrases and `+`/`-` are treated as plain words for now).
- **The cut set.** The executor takes the best `result_cap` matches across ALL types (default 100,
  `NADA_STUDIES_RESULT_CAP`), so choosing a tab never changes which studies are relevant. `found` counts the cut set
  (at most the cap), `truncated` says whether more studies matched, and `search_counts_by_type` describes the cut set;
  `types` narrows `found` and the hits only. Order ties break by `sid`, so paging is deterministic.
- **Other sorts.** A non-relevance sort re-sorts the same cut set inside OpenSearch by pinning it with a `sids`
  filter and reusing the browse query (two requests, no sort-key normalization duplicated in Python); each hit keeps
  its relevance score. `debug` lists every OpenSearch request.
- **Executors take a `SearchJob`** (client, index, query, filters, sort, paging, cap); `EXECUTORS` now serves `browse`
  and `lexical`, so `GET /info` advertises `lexical`. `semantic` and `hybrid` still answer `unsupported_capability`.
- **Verification.** A live test on real OpenSearch covers gibberish, case and accent insensitivity, typos, idno
  matching, boost ranking (a title match beats an abstract match), `minimum_should_match`, filters before the cut and
  `types` after it, the cap, truncation, re-sorting the cut set, and paging. On the real sample: nonsense queries
  return 0, `povrety` and `cencus` still find their studies, and results are small bounded sets.
- **Evaluated in step 8** (see `docs/opensearch-search-evaluation.md`; `AUTO` fuzziness became `AUTO:5,9`, a stopword
  analyzer did not help). Original questions: (1) With `most_fields`, `minimum_should_match` applies per field, so the terms of a
  multi-word query must all be in the same field; NADA's own search behaves the same, but `cross_fields` (which does not
  support fuzziness) or a fusion of both may recall more. (2) The text fields have no stemming (only typo tolerance),
  so "prices" reaches "price" by edit distance, not by stem. (3) Whether the 2<75% threshold and the boosts suit
  real queries; the golden set will say.

## Step 7 findings

- **The vector leg** (`studies_semantic.py`, `studies_search.semantic`): a kNN request on the chunk index, filters applied
  *before* the search (the top `k` come from the filtered set), chunks collapsed to one hit per study, and each document
  study's best passages returned (`{page (1-based), total_pages, score, excerpt}`). The same 13 filters reach it (a
  `FieldMap` maps them to chunk paths); `created` is stamped on chunks at ingest for that reason. `types` is applied
  after the cut on every leg.
- **The policy** (`StudyPolicy`, all `NADA_STUDIES_*` settings): an absolute floor on the vector score
  (default 0.70 after step 8) decides whether there are any semantic matches at all, then a relative cutoff (0.94 of the best after step 8) bounds
  how many. On this OpenSearch (faiss cosine) scores are compressed into about 0.5-1: real queries peaked at
  0.70-0.85 and gibberish at 0.64-0.67 on the sample, so 0.68 separates them there, but with a thin margin.
  **Re-calibrated on the full catalog in step 8:** the sample numbers did not hold (see below).
- **Fusion:** reciprocal rank fusion (rank constant 60), keyword and vector weight both 1.0 after step 8 (it began as 1.0 and
  0.5 so that keyword matches lead); a study found by both legs outranks either alone. Each leg
  contributes its top `window` (200); the fused ranking is cut at `result_cap` (100) across all types. `matched_by`
  says which leg found each study; `score` is the opaque fused score (the vector score in `semantic` mode).
- **`truncated`** is true when the keyword leg matched more studies than its window, or the fused ranking exceeds the cap.
  A study just outside a leg's window contributes no rank from that leg (a documented approximation of windowed fusion).
- **The study index is the gate.** Vector-only hits are looked up in the study index with the full filters: it supplies
  NADA's own idno (the chunk carries the record's schema idno), the type for the counts, and drops studies that are no
  longer indexed or that a filter excludes.
- **Modes and degradation.** `auto` with a query is `hybrid`. If the query cannot be embedded, `auto` answers with keyword
  search and a `semantic_unavailable` warning; an explicit `hybrid`/`semantic` returns `embedding_unavailable` (503).
  `semantic` and `hybrid` are advertised only when embeddings are local (`GET /info` derives it from the config).
  `debug` shows every request, with the query vector replaced by its dimension.
- **Prune-by-`sid`** (decided here): after a study is written, its chunks from earlier runs that are not in the new set
  are deleted (batched, per study), so a re-index cannot leave stale passages. A study that now produces no chunks loses
  all of them but keeps its study document.
- **Verification.** Unit tests for the helpers, fusion, executors (mocked cluster) and route; a live test with a
  deterministic concept-based embedding checks each leg, the fusion order and `matched_by`, semantic-only finds, NADA idno,
  passages, gibberish, filters on both legs (including `created`), `types` and tab counts, the cap, re-sorting, the
  policy knobs and pruning. On the real sample with the real model (`microsoft/harrier-oss-v1-270m`, 640 dims) queries
  behave as intended (see the step 7 summary); the sample has no page-level passages, so passages are verified by the
  unit and live tests only.
- **Open after step 8:** the window, `most_fields` versus other lexical queries, and whether semantic-only results should
  be labelled in the UI.

## Parallel NADA tasks (not blocking)

- Extract additions: subtitle, aliases. Steps 2-3 work without them.
- State reports by id instead of idno. Until then nada-ai sends both.
- The `semantic_search_engine` setting and the new thin NADA driver (after step 7).

## Guardrails while testing

- The scratch OpenSearch instance and scratch nada-ai API must never report state to NADA. Run them with
  `NADA_REPORT_SEARCH_INDEX_STATE_ENABLED=false` and `NADA_RECONCILE_SEARCH_INDEX_ENABLED=false`; NADA's
  `search_index_state` table tracks a single engine.
- Do not touch the existing Qdrant containers (`nada-ai-api-qdrant-dev`, `nada-ai-qdrant-dev`, ports 8020, 6333).

## Deferred (review later)

- **Reminder:** contract auth header (accept `Authorization: Bearer`?) and protecting the unauthenticated `POST /search`.
- **Reminder:** query syntax. For quoted phrases and `+`/`-` operators, decide whether the query still runs
  semantic/hybrid or switches to lexical.
- Cap of 100 for broad keyword queries (`truncated`), and whether unknown `types` should stay a zero-result rather
  than an error.
- **Reminder:** review moving admin routes from `{idno}` to `{sid}`, and the wider admin route cleanup
  (duplicate filter families, `/admin/qdrant/collection` versus `/admin/index`, engine names in paths).
- Variables and citations search, Solr backend, dynamic facets from OpenSearch (`include_facets`), reranker,
  deprecating duplicate admin routes.
