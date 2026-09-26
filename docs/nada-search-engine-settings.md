# NADA search engine settings

How NADA decides which engine answers a search, and how nada-ai fits in. The settings live in NADA (Site
configurations > Search); nada-ai's own environment (`NADA_SEARCH_BACKEND`, `NADA_EMBEDDING_BACKEND`, index names)
describes the deployment and is not a NADA setting.

## Settings

| Setting | Values | Notes |
|---|---|---|
| `search_engine` | `database`, `solr`, `opensearch_native`, `nada_ai` | The engine that serves catalog search. `opensearch_native` is NADA's own OpenSearch integration (its `nada*` indexes), unrelated to nada-ai's index. |
| `nada_ai_url` | URL | Required before `nada_ai` can be selected. |
| `nada_ai_api_key`, `nada_ai_admin_api_key` | secrets | Search calls, and the indexing dashboard's `/admin` calls. Never the same secret. |
| `nada_ai_on_outage` | `database` (default), `error` | What a search does while nada-ai is down. |
| `nada_ai_combine_with_database` | `true`, `false` | Only for a nada-ai that runs Qdrant: also run the database keyword search and show the best semantic matches first. |
| `nada_ai_debug` | `true`, `false` | Attach request/response payloads for signed-in administrators. |
| `nada_ai_timeout`, `nada_ai_breaker_cooldown` | seconds | `application/config/semantic_search.php`. |

There is one engine setting, not one per search. Site settings win over the config files, which only supply defaults.

## What serves what

Studies, variables and citations are each served by `search_engine` when it can be, and by the database when it cannot.
For `nada_ai`, NADA asks nada-ai (`GET /info`, cached for a minute) which engine it runs and which searches it has:

| nada-ai runs | Studies | Variables | Citations |
|---|---|---|---|
| OpenSearch | nada-ai | nada-ai | nada-ai |
| Qdrant | nada-ai | database | database |

Solr and `opensearch_native` serve studies and citations; the catalog's variable view is still served by the database for
them. The settings page shows what serves each search right now.

The engine nada-ai runs is never typed into NADA: with Qdrant, `nada_ai_combine_with_database` picks between the plain
Qdrant driver and the one that combines it with the database search.

## When nada-ai does not answer

| Situation | What happens |
|---|---|
| The engine lacks the search (Qdrant has no citation search) | The database serves it, every time. Not a fallback. |
| A request the index cannot answer (no keyword, unpublished citations, admin-only filters, an unsupported sort or page size) | The database serves that request. |
| nada-ai is down: no connection, timeout, HTTP 5xx except 501, HTTP 429 | `nada_ai_on_outage`: the database, or an error with the reason. |
| nada-ai rejects the request: HTTP 4xx, or 501 (its engine lacks what was asked) | Always an error. A 501 means the site is set to something the running engine cannot do. |

**Outage breaker.** A failure that means nada-ai is down opens a breaker for `nada_ai_breaker_cooldown` seconds (30):
calls fail at once instead of each waiting for its own timeout. After the cooldown one request probes nada-ai; a success
closes the breaker, a failure opens it again. The state is a few entries in NADA's file cache (`cache/`), shared by all
PHP processes on the server; losing it costs a failed attempt or two. `GET /api/admin/semantic/engine_status` returns the
engine, what serves what, nada-ai's capabilities, the breaker state and the fallbacks counted this hour.

## Change tracking

NADA queues catalog changes for the engine that serves them, so an object type is tracked when the engine serving it is
not the database. There is no setting for it. With `nada_ai` and a Qdrant nada-ai, citations are served by the database,
so their changes are not queued.

## Upgrading

`php index.php cli/migrate latest` runs `20260926120001_rename_search_engine_settings`. It renames the stored settings
(`search_provider` values `db`/`mysql`/`mysqli`/`sqlsrv` -> `database`, `solr` -> `solr`, `opensearch` ->
`opensearch_native`, `semantic` -> `nada_ai`; `semantic_search_url|api_key|admin_api_key|debug` ->
`nada_ai_url|api_key|admin_api_key|debug`), turns `semantic_search_engine = qdrant_db` into
`nada_ai_combine_with_database = true`, and drops `semantic_search_engine` and `citation_search_provider`. It is
idempotent and one-way like NADA's other migrations: back up the database first. Run it **before** using the new code:
until it has run, the new names are unset, so the site searches with the database.
Settings kept in `application/config/config.php` (`search_provider`, `semantic_search_*`) must be renamed by hand.
