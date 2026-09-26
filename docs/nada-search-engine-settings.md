# NADA search engine settings

How NADA decides which engine answers a search, and how nada-ai fits in. The settings live in NADA (Site
configurations > Search); nada-ai's own environment (`NADA_SEARCH_BACKEND`, `NADA_EMBEDDING_BACKEND`, index names)
describes the deployment and is not a NADA setting.

## Settings

| Setting | Values | Notes |
|---|---|---|
| `search_engine` | `database`, `solr`, `opensearch`, `nada_ai_opensearch`, `nada_ai_qdrant` | The engine that serves catalog search. `opensearch` is NADA's own OpenSearch integration (its `nada*` indexes), unrelated to nada-ai's index. The two `nada_ai_*` values say which engine nada-ai is expected to run. |
| `nada_ai_url` | URL | Required before a `nada_ai_*` engine can be selected. |
| `nada_ai_api_key`, `nada_ai_admin_api_key` | secrets | Search calls, and the indexing dashboard's `/admin` calls. Never the same secret. |
| `nada_ai_on_outage` | `database` (default), `error` | What a search does while nada-ai is down. |
| `nada_ai_debug` | `true`, `false` | Attach request/response payloads for signed-in administrators. |
| `nada_ai_timeout`, `nada_ai_breaker_cooldown` | seconds | `application/config/semantic_search.php`. |

There is one engine setting, not one per search. Site settings win over the config files, which only supply defaults.

In Site configurations > Search each engine is a collapsible section (Database, Solr, OpenSearch, nada-ai); the active one is
marked, and "Use this engine" selects one (saved with the section's Save button). nada-ai is one section for both
`nada_ai_*` values: a choice of the engine it runs (OpenSearch or Qdrant), then the connection, the outage behaviour and an
Advanced fold, so the settings both engines share appear once.

## What each engine does

| `search_engine` | Studies | Variables | Citations |
|---|---|---|---|
| `database` | database | database | database |
| `solr` | Solr | database | Solr |
| `opensearch` | NADA's OpenSearch | database | NADA's OpenSearch |
| `nada_ai_opensearch` | nada-ai's OpenSearch (order, counts, tab counts) | nada-ai | nada-ai |
| `nada_ai_qdrant` | Qdrant's best semantic matches first, then the database keyword results | database | database |

The database always loads the rows of a page and serves the sidebar facets. With `nada_ai_qdrant` it also serves the
totals and tab counts. NADA only reads which searches nada-ai has from `GET /info` (cached for a minute), so a nada-ai
that gains a search is used without a NADA change. Solr and NADA's OpenSearch have a variable search, but the catalog's
variable view is still served by the database for them.

**nada-ai must run what the setting says.** NADA reads the engine nada-ai reports (`/info`). If it differs from the
`nada_ai_*` value, catalog searches fail with a message naming both, and the settings page shows a warning; NADA does not
guess. This also catches a nada-ai that was restarted on another backend, whose index would be empty or of the wrong kind.

## When nada-ai does not answer

| Situation | What happens |
|---|---|
| The engine lacks the search (nada-ai with Qdrant has no citation search) | The database serves it, every time. Not a fallback. |
| A request the index cannot answer (no keyword, unpublished citations, admin-only filters, an unsupported sort or page size) | The database serves that request. |
| nada-ai is down: no connection, timeout, HTTP 5xx except 501, HTTP 429 | `nada_ai_on_outage`: the database, or an error with the reason. |
| nada-ai rejects the request: HTTP 4xx, or 501 (its engine lacks what was asked) | Always an error. A 501 means the site is set to something the running engine cannot do. |

**Outage breaker.** A failure that means nada-ai is down opens a breaker for `nada_ai_breaker_cooldown` seconds (30):
calls fail at once instead of each waiting for its own timeout. After the cooldown one request probes nada-ai; a success
closes the breaker, a failure opens it again. The state is a few entries in NADA's file cache (`cache/`), shared by all
PHP processes on the server; losing it costs a failed attempt or two. `GET /api/admin/semantic/engine_status` returns the
engine, what serves what, nada-ai's capabilities, the breaker state and the fallbacks counted this hour. The semantic search
dashboard shows it: a Search engine card on the Overview page, and a banner on every page while nada-ai runs another engine
than the setting or is not answering.

## Change tracking

NADA queues catalog changes for the engine that serves them, so an object type is tracked when the engine serving it is
not the database. There is no setting for it. With `nada_ai_qdrant`, citations are served by the database, so their
changes are not queued.

## Upgrading

`php index.php cli/migrate latest` runs `20260926120001_rename_search_engine_settings`. It renames the stored settings
(`search_provider` values `db`/`mysql`/`mysqli`/`sqlsrv` -> `database`, `solr` -> `solr`, `opensearch` -> `opensearch`,
`semantic` -> `nada_ai_opensearch` when `semantic_search_engine` was `opensearch`, otherwise `nada_ai_qdrant`;
`semantic_search_url|api_key|admin_api_key|debug` -> `nada_ai_url|api_key|admin_api_key|debug`) and drops
`semantic_search_engine` and `citation_search_provider`. The plain Qdrant driver (`semantic_search_engine = qdrant`) no
longer exists: those sites get `nada_ai_qdrant`, which is the driver that combines Qdrant with the database keyword
search (what `qdrant_db` was). The migration is idempotent and one-way like NADA's other migrations: back up the
database first. Run it **before** using the new code: until it has run, the new names are unset, so the site searches
with the database. Settings kept in `application/config/config.php` (`search_provider`, `semantic_search_*`) must be
renamed by hand.
