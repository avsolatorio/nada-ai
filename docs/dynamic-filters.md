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

## IHSN metadata-extract API

Fetch filters from the IHSN admin metadata-extract API and sync them in one step:

```bash
# Single study
uv run python -m nada_ai.filters.cli sync-from-ihsn \
  --idno=RWA_NISR_DOC_2025_CPI-MR_MAY_FR_V1

# Paginate all studies (optional --limit for testing)
uv run python -m nada_ai.filters.cli sync-from-ihsn --all --page-size=100

# Fetch only (no index write)
uv run python -m nada_ai.filters.cli fetch-from-ihsn --all --out=records.json

# Preview what would sync
uv run python -m nada_ai.filters.cli sync-from-ihsn --all --dry-run --limit=5
```

Configure auth in `.env` if the host requires it:

```env
NADA_IHSN_METADATA_EXTRACT_BASE_URL=https://training.ihsn.org/index.php/api/admin/search-metadata-extract
NADA_IHSN_API_KEY=your-key
# or
NADA_IHSN_AUTH_BEARER=your-token
# or
NADA_IHSN_AUTH_COOKIE=session=...
```

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

Fixed keys (`type`, `idno`, `geographies`, …) query legacy metadata paths. Any other key queries `metadata.filter_fields` via nested conditions.

## Facets

Dynamic facet keys are listed in [`config/dynamic_filter_facets.json`](../config/dynamic_filter_facets.json). Override path with `NADA_DYNAMIC_FILTER_FACETS_PATH`.

When `include_facets=true`, the response includes static facets plus registered dynamic facets.

## Operations

### Qdrant

Payload indexes are created for:

- `metadata.filter_fields[].key`
- `metadata.filter_fields[].value`

Run `ensure-indexes` after upgrading an existing collection.

### OpenSearch

Nested mapping for `metadata.filter_fields` is added via `ensure-indexes` (`put_mapping`). Existing indices may require reindex if mapping update is rejected.
