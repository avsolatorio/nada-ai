# Citations search contract

Companion to `studies-search-contract.md` and `variables-search-contract.md`. The standard citation search is
`POST /citations/search`: **lexical only** (a citation is a short bibliographic record, so keyword matching does the
job and nothing is embedded), OpenSearch only. It returns ordered citation ids with a few display fields; NADA hydrates
the full rows (authors, survey counts, ...) from its own database, as it does for studies.

## 1. Request

```json
{
  "query": "poverty",
  "filters": { "ctypes": ["book"], "year_from": 2000, "year_to": 2010 },
  "sort": "relevance",
  "order": "asc",
  "limit": 15,
  "offset": 0
}
```

| Field | Notes |
|---|---|
| `query` | Required, 1-500 characters. Matched against title (x5), authors (x3), subtitle (x2), keywords (x2), abstract and notes. An exact, case-folded DOI match outranks any text match. |
| `filters.ctypes` | Citation types, any-of. |
| `filters.year_from` / `year_to` | Publication year, inclusive; either may be omitted. `year_from` after `year_to` is `invalid_filter_value`. |
| `sort` | `relevance` (default), `title` or `year`. |
| `order` | `asc` (default) or `desc`; applies to `title` and `year` only. Citations with no year sort last either way. |
| `limit` / `offset` | `limit` 1-100 (default 15); `offset + limit` must not exceed 10,000. |

Only **published** citations are ever searched; that is not a filter. Unknown filter keys are `unknown_filter`
(422), listing the supported ones. Errors use the study search's envelope and codes (`invalid_request`,
`invalid_filter_value`, `unknown_filter`, `offset_out_of_range`, `unsupported_capability` (501 on Qdrant),
`index_not_ready` (503 before the first sync), `backend_unavailable`, `query_rejected`, `unauthorized`, `forbidden`,
`rate_limited`).

## 2. Response

`{engine, found, limit, offset, truncated, hits, applied, timing_ms}`; each hit is
`{rank, citation_id, uuid, title, authors, ctype, pub_year, doi, score}`. `authors` is the flattened name list
(`"Jane Doe; Ann B Smith"`). `citation_id` is NADA's `citations.id`.

`GET /info` reports `capabilities.citations_search: true` on OpenSearch.

## 3. What is not supported

Filters NADA's own database search has and this one does not: flag, user, url status, has notes, no survey attached and
the repository scope. They are admin-list filters that are not in the index. NADA's driver for this search says so and
does not send them (NADA serves citations from the engine its `search_engine` setting names, and from the database when that engine has no citation search).

## 4. Indexing

- **The whole catalog:** `POST /admin/citations/sync` (role `write`, OpenSearch only) with an empty body runs a
  background job (kind `index_citations`) that walks NADA's paged `/api/admin/search-metadata-extract/citations` route
  (keyset on the citation id, page cap `search_metadata_extract_citations_max_limit`, 500). Idempotent: `_id` is the
  citation id. It adds and replaces; it does not remove.
- **Some citations:** `{"ids": [3, 7]}` syncs those; each is rewritten, or removed when NADA no longer has it. An empty
  list is a 400, not "everything".
- **The change queue:** NADA's queue already tracks citations (`object_type = citation`). `reconcile_once`, the
  scheduler and the Change queue page now process them together with studies: an edit or publish rewrites that citation,
  a delete removes it. A read failure leaves the existing document alone and acks the item as failed. On Qdrant there
  is no citation index, so the item is acked without work.
- **Totals:** `GET /admin/citations/stats` returns `{index, exists, citations, published}`; `exists: false` before the
  first sync is a normal state.
- **Lifecycle:** the citation index is independent of the study indexes. Recreating or dropping the study index does
  **not** touch it (citations do not depend on studies, and a rebuild does not re-ingest them).
- **Index name:** `<index_name>-citations`, or `NADA_CITATIONS_INDEX_NAME`.
- **Not tracked:** the diff/stale views and search-index state reporting still cover studies only.
