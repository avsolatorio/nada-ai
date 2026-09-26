# Pending items

Open work and decisions after the variables, citations and publish-in-place changes (2026-09-25). Grouped by what
blocks each item. Where it lives: **nada-ai** (this repo), **nada-semantic** (NADA, branch `develop-semantic-ui`) or
**ai4data** (the discovery library, pinned by git rev in `pyproject.toml`).

## 1. Waiting on a decision or an action

| # | Item | Notes |
|---|---|---|
| 1 | ~~Commit the staged changes~~ | Done 2026-09-25: nada-ai `9883ae4` (branch `feat/opensearch`), nada-semantic `191fac4b` (branch `develop-semantic-ui`), ai4data `6d2e13f`. None pushed except ai4data (fork). nada-ai and nada-semantic have not been pushed to their remotes. |
| 2 | Push nada-semantic to `ihsn/nada` | Only when asked. |
| 3 | Stray `nada-os-baseline-variables` index on the Homebrew OpenSearch (`localhost:9200`) | Written by a variables sync while nada-ai pointed at the wrong OpenSearch. Unrelated to the scratch instance (Docker, port 9201). Drop it or leave it. |
| 4 | Retest a status flip end to end | `rwa-nisr-russ-2012-v1` and `CES-EC-2014-06` are still pending as full in the change queue. Process them from the Change queue page, flip a status, and check the row queues as `upsert_partial`. Not run since NADA's `atomic`/`publish` mapping was restored. |
| 5 | Click through the new Search section in Site configurations | Verified through the APIs, the migration on the dev database and the search paths (nada-ai OpenSearch and Qdrant); the page itself was compiled and linted but not opened in a browser. |
| 5a | Run `php index.php cli/migrate latest` on other NADA installs before deploying the new code | Renames the search settings; see `nada-search-engine-settings.md`. On the dev database only that migration's `up()` was applied; the earlier pending `20260918120001` (purges 340 `deleted` rows of `search_index_state`) is not applied. |
| 6 | Automatic queue processing | `NADA_RECONCILE_SEARCH_INDEX_ENABLED` is off (default, and in the scratch script). Pending rows wait for "Process pending" on the Change queue page. |
| 7 | Diff page "Reconcile now" wording | An info note now says it does not process the change queue. The button label is unchanged. |

## 2. Known issues, not investigated

| # | Item | Notes |
|---|---|---|
| 8 | 15 studies fail every diff reconcile | An idno that is a URL, studies with empty content, a metadata fetch error. Predates the variables work. |
| 9 | nada-ai `.env` uses `host.docker.internal` | Right for Docker, unresolvable on the host, so a host run needs `AI4DATA_METADATA_CATALOG_URL=http://localhost/...`. `scripts/local-opensearch.sh` sets everything itself. |
| 10 | Citations are not in the diff/stale views or state reporting | Those cover studies only. There is also no per-citation gaps list like the one for variables. |
| 11 | New filters are not tracked | The filter field appears in the index on the first indexed study (dynamic mapping) and search accepts `fq_<name>`. Gaps: no queue signal when a facet is created or deleted (a deleted facet's values stay in documents until each study is reindexed); no view of which filter keys the index holds; the facet registry is a local file on the nada-ai host, not in NADA or the state table; the mapping has a 1,000-field default limit. |

## 3. Decided against or not built

| # | Item | Notes |
|---|---|---|
| 12 | Metadata-only publish/unpublish as its own path | Superseded: NADA sends `upsert_partial` for any options change (`publish` and `atomic`), and nada-ai applies it in place (`apply_study_options_op`): study document, chunk `filter_facets`, variables' `published`. No metadata read, no embedding. |
| 13 | Skip variable sync when a study's `varcount` is 0 | Not implemented. |
| 14 | Per-idno "variables only" option on the Index page | Not added; the Variables strip and its studies-to-sync table cover it. |
| 15 | Per-variable or per-study variable state tracking | Dropped by decision. Totals plus a per-study count comparison are used instead. |

## 4. Parked, not started

| # | Item | Notes |
|---|---|---|
| 16 | `search_index_state` has no engine tracking | Switching engines (OpenSearch, Qdrant, database) cannot be detected from the state table, so a full reindex or reconcile cannot be triggered by it. |
| 17 | Endpoints that need HTTP DELETE or PATCH | The deployment environment blocks both. Review the affected endpoints; do not add new ones without flagging it. |
| 18 | Admin routes keyed by `{idno}` | Review moving them to `{sid}` (see `studies-search-contract.md` §10). |
| 19 | Generic collection-info endpoint | |
| 20 | Solr backend | |
| 21 | OpenSearch quality gaps | Noise, overlap, alias, scale, and `qdrant_db` parity, explicitly deferred. |

## 4a. Search engine clean-up: what is left

| # | Item | Notes |
|---|---|---|
| 22 | ~~Admin summary and outage banner (phase 4)~~ | Done: a Search engine card on the Overview page (engine, what serves studies/variables/citations, what nada-ai runs, fallbacks this hour) and a banner on every dashboard page when nada-ai runs another engine than the setting or is not answering. `served_by` / `fallback_reason` are still not in search results. |
| 23 | Variable view for Solr and native OpenSearch | Still served by the database (`NATIVE_ENGINES_SERVE_VARIABLES = false`). Decided to leave for now. |

## 5. Behaviours worth remembering

- The variable index is emptied by a study-index recreate; a microdata run or **Index variables** refills it. The
  citation index is independent and is not touched by a recreate.
- Unchanged chunks are not re-embedded on the local backend: a chunk id hashes its text, so its stored vector is
  reused when the index model matches and the run is not forced.
- Citation search falls back to the database in NADA for what the index cannot answer (no keyword, unpublished,
  collection scope, admin-only filters, sorts other than title/year/relevance, pages over 100) and when nada-ai is
  unreachable or fails on its side. See `citations-search-contract.md`.
- The variable view of the public catalog now uses nada-ai's variable search when the provider is `semantic`.
- `npm run build` in `nada-semantic/frontend` is needed after frontend changes; `frontend/dist` is committed.
