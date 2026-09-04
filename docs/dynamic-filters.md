# Dynamic filters

External catalog filters are stored in **`metadata.filter_fields`** as an array of `{key, value[]}` objects. Existing ai4data metadata (`metadata.year_start`, `metadata.geographies`, etc.) is not modified.

## Input format

Single record (CLI or API):

```json
{
  "idno": "DOC-12345",
  "filters": {
    "doctype": 1,
    "countries": [181],
    "regions": [7, 9],
    "tags": []
  }
}
```

Normalization rules:

- `null`, empty strings, and empty lists are dropped
- scalars become single-element lists
- all values are stored as strings (e.g. `181` → `"181"`)

## Sync tools

```bash
# Single idno
uv run python -m nada_ai.filters.cli sync --idno=DOC-123 --filters-file=filters.json

# Batch: [{idno, filters}, ...]
uv run python -m nada_ai.filters.cli sync-batch --file=batch.json

# Read back stored filter_fields
uv run python -m nada_ai.filters.cli get --idno=DOC-123

# Ensure indexes / mapping
uv run python -m nada_ai.filters.cli ensure-indexes
```

Admin API (when `NADA_ADMIN_API_KEY` is set):

- `POST /admin/filters/sync` — body `{records: [{idno, filters}]}`
- `GET /admin/filters/{idno}`
- `POST /admin/filters/ensure-indexes`

Sync updates **all points** sharing the same `metadata.idno` (all langdoc chunks).

## NADA metadata-extract API

Fetch filters from your NADA instance's admin metadata-extract API and sync them in
one step (the examples below use IHSN's training instance, but any NADA instance's
extract API works the same way — see `NADA_METADATA_EXTRACT_BASE_URL` below):

```bash
# Single study
uv run python -m nada_ai.filters.cli sync-from-extract \
  --idno=RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1

# Paginate all studies (optional --limit for testing)
uv run python -m nada_ai.filters.cli sync-from-extract --all --page-size=100

# Fetch only (no index write)
uv run python -m nada_ai.filters.cli fetch-from-extract --all --out=records.json

# Preview what would sync
uv run python -m nada_ai.filters.cli sync-from-extract --all --dry-run --limit=5
```

Configure auth in `.env` — this uses the same `AI4DATA_METADATA_CATALOG_*` admin
credential every other admin-API feature uses (search-index reconciliation,
extract-mode content ingest), not a separate one:

```env
AI4DATA_METADATA_CATALOG_URL=https://training.ihsn.org/index.php
AI4DATA_METADATA_CATALOG_EXTRACT_PATH=api/admin/search-metadata-extract
AI4DATA_METADATA_CATALOG_X_API_KEY=your-key
# or
AI4DATA_METADATA_CATALOG_AUTH_BEARER=your-token
# or
AI4DATA_METADATA_CATALOG_COOKIES=session=...
```

Only set `NADA_METADATA_EXTRACT_BASE_URL` if your extract API lives at a
different host/path than `{AI4DATA_METADATA_CATALOG_URL}/{AI4DATA_METADATA_CATALOG_EXTRACT_PATH}`
derives — see `nada_ai.nada.admin_auth` for why credentials have no separate
override.

Query flags (defaults match typical filter-only extraction):

- `--include_admin_metadata=true` (default)
- `--include_metadata=false` (default) — set `true` to include full study metadata in API responses

## Search

Pass dynamic keys alongside fixed filters in `SearchRequest.filters`:

```json
{
  "query": "poverty",
  "filters": {
    "type": "document",
    "countries": [181],
    "doctype": 1
  }
}
```

Fixed keys (`type`, `idno`, `geographies`, …) query legacy metadata paths. On **Qdrant**, any other key queries **`metadata.filter_facets.<key>`** (flat, indexed). On **OpenSearch**, dynamic keys still use nested **`metadata.filter_fields`**.

## Facets

Dynamic facet keys are listed in [`config/dynamic_filter_facets.json`](../config/dynamic_filter_facets.json). Override path with `NADA_DYNAMIC_FILTER_FACETS_PATH`. The file has two arrays:

```json
{"facetable": ["doctype", "countries", ...], "excluded": []}
```

**`facetable` keys don't require manual curation anymore.** Every `sync_filters_for_idno` call (CLI, admin API, or batch sync) auto-registers any key present in the incoming filters dict that isn't already known — and on Qdrant, immediately creates its payload index — so a new key NADA starts sending shows up as a working facet on its very next sync, with no `POST /admin/facets` step required. This is safe because NADA's `filters` field (verified live against `nada-demo.ihsn.org`) is itself a stable, purpose-built projection — not arbitrary metadata — so trusting its keys directly is reasonable.

`excluded` is a deny-list: calling `DELETE /admin/facets/{key}` (or `remove_facet_keys`) both removes a key from `facetable` *and* adds it to `excluded`, so auto-registration won't resurrect it the next time NADA sends data containing it. `POST /admin/facets` (`add_facet_keys`) always wins over a prior exclusion — an explicit add clears the key from `excluded` too.

When `include_facets=true`, the response includes static facets plus registered dynamic facets.

## Operations

### Qdrant

Sync writes both shapes on each point:

- `metadata.filter_fields` — array of `{key, value[]}` (OpenSearch parity, backfill source)
- `metadata.filter_facets` — flat map `{key: [value, ...]}` (Qdrant filter + facet paths)

Payload indexes are created on **`metadata.filter_facets.<key>`** for each facetable key (filter + facet):

```bash
NADA_SEARCH_BACKEND=qdrant uv run python -m nada_ai.filters.cli ensure-indexes
```

Migrate existing points that only have `filter_fields`:

```bash
NADA_SEARCH_BACKEND=qdrant uv run python -m nada_ai.filters.cli backfill-facets
```

Or re-sync from the metadata-extract API (writes both fields):

```bash
NADA_SEARCH_BACKEND=qdrant uv run python -m nada_ai.filters.cli sync-from-extract --all --page-size=100
```

The search API auto-creates missing indexes on first dynamic facet request; running `ensure-indexes` explicitly after deploy is recommended.

### OpenSearch

Nested mapping for `metadata.filter_fields` is added via `ensure-indexes` (`put_mapping`). Existing indices may require reindex if mapping update is rejected.
