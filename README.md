# nada-ai (`nada_ai`)

Python package for **NADA AI**: ingest [NADA / Data Compass–style](https://data-compass.ihsn.org/) metadata via **`ai4data.discovery`**, search it (keyword, k-NN vector, hybrid) over **OpenSearch or Qdrant**, and expose catalog search + timeseries analytics to LLM agents via an **MCP server** (tools, interactive apps, resources, prompts).

**New here? Read the [developer guide](docs/GUIDE.md)** — architecture, semantic search, the MCP server, NADA connectivity, configuration reference, and deployment, all in one place.

## Requirements

- Python **3.11+**
- [**uv**](https://docs.astral.sh/uv/) recommended
- **`ai4data[discovery]`** resolved from **`[tool.uv.sources]`** — pinned to a **git revision** ([`avsolatorio/ai4data`](https://github.com/avsolatorio/ai4data.git), see `pyproject.toml`). Update the `rev` hash there (and run **`uv lock`**) when you want a newer discovery stack. To hack on a **local checkout** instead, temporarily override sources (see uv docs / `tool.uv.sources`) or clone beside the repo and point `path`.

## Install

From this directory:

```bash
cd nada-ai
uv sync --all-groups
# Local SentenceTransformer embeddings (default search/ingest path):
uv sync --extra local
```

For **Amazon OpenSearch / IAM SigV4**:

```bash
uv sync --extra aws
```

For **Qdrant** vector search (recommended local stack — see guide below):

```bash
uv sync --extra local --extra qdrant
```

## Guides

| Guide | Description |
|-------|-------------|
| **[Developer guide](docs/GUIDE.md)** | Full reference: architecture, semantic search, ingestion, the MCP server (tools/apps/resources/prompts), NADA connectivity, every config variable, deployment, observability |
| **[Qdrant pipeline guide](docs/qdrant-pipeline-guide.md)** | End-to-end catalog ingest and search with Qdrant — **host ingest** and **full Docker** setups, metadata-extract catalog, verification, troubleshooting |
| [Dynamic filters](docs/dynamic-filters.md) | Sync catalog filters from a NADA instance's metadata-extract API into the index and search by facet keys |

## Configuration (`NADA_*`)

Settings use the **`NADA_`** prefix (see `nada_ai.settings`). **[`.env.example`](.env.example) is the authoritative, fully-documented list of every variable** (search backend, embeddings, admin auth/RBAC, rate limiting, logging, the standalone MCP server, and the `AI4DATA_*` discovery/catalog config) — copy it to `.env` and edit. Highlights:

| Variable | Purpose |
|----------|---------|
| `NADA_SEARCH_BACKEND` | `qdrant` (default) or `opensearch` — see [Qdrant pipeline guide](docs/qdrant-pipeline-guide.md) |
| `NADA_QDRANT_URL` | Qdrant HTTP URL (default `http://localhost:6333`; compose uses `http://qdrant-nada:6333`) |
| `NADA_QDRANT_COLLECTION_NAME` | Qdrant collection (defaults to `NADA_INDEX_NAME`) |
| `NADA_OPENSEARCH_URL` | Cluster URL (default `http://localhost:9200`) |
| `NADA_INDEX_NAME` | Index name (default `nada-metadata`) |
| `NADA_EMBEDDING_BACKEND` | `local` (default) or `opensearch_ml` |
| `NADA_EMBEDDING_MODEL_ID` | Hugging Face model id for `local` |
| `NADA_QUERY_PROMPT_NAME` | Named asymmetric prompt for `SentenceTransformer.encode(..., prompt_name=...)` when `NADA_QUERY_PROMPT` is unset |
| `NADA_QUERY_PROMPT` | Optional literal prefix for `encode(..., prompt=...)`; when set, overrides `NADA_QUERY_PROMPT_NAME` for query vectors |
| `NADA_OPENSEARCH_AUTH_MODE` | `basic` or `aws_sigv4` |
| `NADA_AWS_REGION` | Required for SigV4 when not implicit from boto3 |

### Admin auth, RBAC, audit trail, rate limiting

Every admin/catalog/facets/webhook/job route requires a principal with a minimum role (`read` / `write` / `admin`), resolved from either of two credential sources:

- **`NADA_ADMIN_API_KEY`** — a legacy super-admin key, checked directly against the environment (not part of `Settings`). Any caller presenting this value as `X-NADA-Admin-Key` gets role `admin`.
- **Per-caller API keys** — issue scoped, revocable keys via `POST /admin/keys` (see `nada_ai.app.keys_admin`); each is stored hashed at `NADA_API_KEYS_PATH` (default `config/api_keys.json`).

If neither `NADA_ADMIN_API_KEY` nor any stored key exists, the server runs fully **unauthenticated** with a loud startup warning — convenient for local dev, never acceptable for anything reachable over a network. Every mutating action is recorded to an append-only audit trail (`NADA_AUDIT_LOG_PATH`, default `config/audit.log`), queryable via `GET /admin/audit`. Public search endpoints (`/search`, `/recommendations`, `/search/explain`, PDF preview) are rate-limited per caller via `NADA_RATE_LIMIT_SEARCH_PER_MINUTE` (default 120/min, 0 disables).

**Discovery caches** (standalone `ai4data`):

| Variable | Purpose |
|----------|---------|
| `AI4DATA_DISCOVERY_DATA_PATH` | Writable directory for discovery caches and bundled metadata helpers (set in Docker and CLI jobs) |
| `AI4DATA_METADATA_CATALOG_URL` | NADA catalog base URL (default Data Compass) |
| `AI4DATA_METADATA_CATALOG_EXTRACT_PATH` | When set, use bulk `search-metadata-extract/studies` instead of catalog search + JSON — [details](docs/qdrant-pipeline-guide.md#configuration) |

Optional: call `init_discovery_paths(Path(...))` from `ai4data.discovery.config` at startup if you prefer code over env.

## Search backends

### Qdrant

Use **`docker-compose.qdrant.yml`** and follow **[docs/qdrant-pipeline-guide.md](docs/qdrant-pipeline-guide.md)** for host vs Docker ingest, catalog configuration, and search verification.

Quick start:

```bash
mkdir -p data/nada-discovery
docker compose -f docker-compose.qdrant.yml up --build -d
docker exec nada-ai-api-qdrant-dev python -m nada_ai.ingest.cli create_index
docker exec nada-ai-api-qdrant-dev python -m nada_ai.ingest.cli index_from_catalog --catalog_type=document --limit=10
curl -s -X POST http://localhost:8020/health/embeddings/warmup
```

### OpenSearch

The FastAPI app exposes **`POST /search`** (keyword, vector, hybrid), **`GET /demo`**, health routes, and **admin/ingest** routes that mirror the CLI. Search query DSL lives under **`nada_ai.search.backend.opensearch`**.

#### Version compatibility

Target server: **OpenSearch 3.x** (e.g. **3.6 LTS**). The Python client is **`opensearch-py` 3.x**, aligned with supported server versions per [upstream compatibility](https://github.com/opensearch-project/opensearch-py/blob/main/COMPATIBILITY.md).

### Operations cheat sheet

Assume `cd nada-ai` unless noted.

#### Docker Compose: build, start, stop

```bash
# Start OpenSearch + FastAPI API (detached)
docker compose -f docker-compose.opensearch.yml up -d

# Rebuild images after Dockerfile / dependency changes
docker compose -f docker-compose.opensearch.yml up -d --build

# Recreate only the API container (e.g. after editing compose environment)
docker compose -f docker-compose.opensearch.yml up -d --force-recreate nada-ai-api

# Stop and remove containers (add -v to drop dev volumes — wipes index data if stored in a volume)
docker compose -f docker-compose.opensearch.yml down

docker compose -f docker-compose.opensearch.yml ps
docker logs nada-ai-opensearch-dev 2>&1 | tail -50
docker logs nada-ai-api-dev 2>&1 | tail -50
```

#### Dev overlay (bind-mount `src`, no rebuild for Python edits)

Use **`docker-compose.dev.yml`** together with the base file. It bind-mounts **`./src`**, sets **`PYTHONPATH=/workspace/src`** so imports use the mount (not only the wheel under `.venv`), runs **uvicorn `--reload`**, and loads **`.env`** when it exists (`required: false`).

```bash
docker compose -f docker-compose.opensearch.yml -f docker-compose.dev.yml up --build -d
# logs: docker logs -f nada-ai-api-dev
```

Optional: set **`COMPOSE_FILE`** so plain **`docker compose up`** uses both files (Unix uses `:` between paths; Git Bash/WSL same; Windows PowerShell uses `;`).

Rebuild **`nada-ai-api`** when **`Dockerfile`**, **`pyproject.toml`**, or **`uv.lock`** change; day-to-day Python changes under **`src/`** only need a container restart if you are *not* using this overlay (with the overlay, save files and let reload pick them up).

OpenSearch: **http://localhost:9200** · API: **http://localhost:8020**. First boot of OpenSearch can take ~30–60s until `curl -s http://localhost:9200/` returns JSON.

#### Health, embeddings readiness, warmup

From the **host** (ports published by compose):

```bash
curl -s http://localhost:9200/ | head
curl -s http://localhost:8020/health
curl -s http://localhost:8020/health/embeddings
curl -s -X POST http://localhost:8020/health/embeddings/warmup
```

From **inside** the API container (`opensearch-nada` is the Compose service name on the Docker network; the API listens on **`127.0.0.1:8020`** inside that container):

```bash
docker exec nada-ai-api-dev curl -s http://opensearch-nada:9200/ | head
docker exec nada-ai-api-dev curl -s http://127.0.0.1:8020/health
docker exec nada-ai-api-dev curl -s http://127.0.0.1:8020/health/embeddings
docker exec nada-ai-api-dev curl -s -X POST http://127.0.0.1:8020/health/embeddings/warmup
```

Use **warmup** after restarts so the first vector/hybrid search is not blocked by model load. `GET /health/embeddings` reports `not_initialized` until warmup or the first vector/hybrid request.

#### Create the OpenSearch index

From the host (same `uv` env as ingest; `NADA_OPENSEARCH_URL` must point at the cluster, e.g. `http://localhost:9200`):

```bash
uv run python -m nada_ai.ingest.cli create_index
# Destructive: drop and recreate
uv run python -m nada_ai.ingest.cli create_index --recreate=True
```

Inside the API container (`NADA_OPENSEARCH_URL` defaults to `http://opensearch-nada:9200`):

```bash
docker exec nada-ai-api-dev python -m nada_ai.ingest.cli create_index
```

For **`embedding_backend=local`**, the index **`knn_vector`** dimension must match the SentenceTransformer model used at ingest and search time. Mismatches produce client errors on vector/hybrid search.

#### Ingest: host machine

Point at Docker OpenSearch on the host and set a writable discovery cache (matches compose’s `./data/nada-discovery`):

```bash
export NADA_OPENSEARCH_URL=http://localhost:9200
export NADA_INDEX_NAME=nada-metadata
export NADA_EMBEDDING_BACKEND=local
export NADA_EMBEDDING_MODEL_ID=microsoft/harrier-oss-v1-270m
export NADA_QUERY_PROMPT_NAME=web_search_query
export AI4DATA_DISCOVERY_DATA_PATH="$(pwd)/data/nada-discovery"
# Optional on Apple Silicon:
# export NADA_EMBEDDING_DEVICE=mps

uv run python -m nada_ai.ingest.cli index_from_catalog --catalog_type=timeseries
```

#### Ingest: inside the API container

Compose mounts **`./data/nada-discovery`** → **`/workspace/data/nada-discovery`** and sets **`AI4DATA_DISCOVERY_DATA_PATH`** accordingly.

```bash
docker exec -it nada-ai-api-dev \
  python -m nada_ai.ingest.cli index_from_catalog --catalog_type=timeseries
```

#### Docker disk pressure (when OpenSearch exits with “no space left on device”)

```bash
docker system df
docker container prune -f
docker builder prune -f
```

Then `docker compose -f docker-compose.opensearch.yml up -d` again.

#### 403 `index_create_block_exception` / “cluster create-index blocked”

Usually **disk watermarks**: OpenSearch blocks new indices when the node thinks the disk is too full (common with Docker Desktop’s disk image). The compose file sets `cluster.routing.allocation.disk.threshold_enabled=false` for local dev.

If you still see 403 on `indices.create` after upgrading the compose file, **recreate** the OpenSearch container so settings apply, or clear blocks on a running node:

```bash
curl -s -X PUT "http://localhost:9200/_cluster/settings" \
  -H 'Content-Type: application/json' \
  -d '{"persistent":{"cluster.routing.allocation.disk.threshold_enabled":false,"cluster.blocks.read_only_allow_delete":false}}'
```

Freeing host disk space also helps. As a last resort: `docker compose -f docker-compose.opensearch.yml down` (add `-v` only if you accept wiping dev index data).

## Docker Compose (API + OpenSearch)

Build context is the **`nada-ai`** repo root. **`ai4data`** is fetched from Git at the **`rev`** pinned in **`pyproject.toml`** / **`uv.lock`** during **`uv sync --frozen`** (no sibling `ai4data` folder required).

From `nada-ai/`:

```bash
mkdir -p data/nada-discovery
docker compose -f docker-compose.opensearch.yml build
docker compose -f docker-compose.opensearch.yml up -d
```

The API listens on **8020**, OpenSearch on **9200**. Compose sets `AI4DATA_DISCOVERY_DATA_PATH` and mounts `./data/nada-discovery`. Model caches can use `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` volumes as in `docker-compose.opensearch.yml`.

## CLI (`nada_ai.ingest`)

```bash
uv run python -m nada_ai.ingest.cli create_index
uv run python -m nada_ai.ingest.cli index --idnos=WB_123 --metadata_type=indicator
uv run python -m nada_ai.ingest.cli index_from_catalog --catalog_type=timeseries --limit=50
```

For **`opensearch_ml`**, register a model in ML Commons, set `NADA_OPENSEARCH_ML_MODEL_ID` and `NADA_OPENSEARCH_ML_EMBEDDING_DIMENSION`, then:

```bash
uv run python -m nada_ai.ingest.cli setup_ingest_pipeline
```

## Admin / ingest endpoints (HTTP)

The 4 CLI commands are exposed as **non-blocking** HTTP endpoints. Each `POST` returns `202 Accepted` with a `job_id` and the work continues in the background. Submitting the same operation while it is still running returns **`409 Conflict`** with the existing job (single-flight by request key) — *the same operation never runs twice in parallel*. Poll `GET /jobs/{id}` for status.

Auth: when `NADA_ADMIN_API_KEY` is set, all `/admin/*` endpoints require header `X-NADA-Admin-Key: <value>`. `/jobs*` endpoints are always open so progress can be polled.

### CLI mirrors

| Endpoint | CLI equivalent | Single-flight key |
|---|---|---|
| `POST /admin/index` `{recreate?: bool}` | `cli create_index --recreate=...` | `create_index` |
| `POST /admin/setup-ingest-pipeline` | `cli setup_ingest_pipeline` | `setup_ingest_pipeline` |
| `POST /admin/ingest/by-ids` `{idnos: [...], metadata_type, force?, recreate_index?, buffer_size?}` | `cli index --idnos=...` | `index:{metadata_type}:{sha1(sorted idnos)}` |
| `POST /admin/ingest/from-catalog` `{catalog_type, ps?, limit?, force?, recreate_index?, buffer_size?}` | `cli index_from_catalog --catalog_type=...` | `index_from_catalog:{catalog_type}` |

Different keys can run concurrently (e.g. `index_from_catalog timeseries` and `index_from_catalog document`); identical keys cannot.

### Job control

| Endpoint | Purpose |
|---|---|
| `GET /jobs?status=...&limit=...` | List recent jobs, optionally filtered by status |
| `GET /jobs/{id}` | Single job snapshot (status, started_at, finished_at, result, error, progress) |
| `DELETE /jobs/{id}` | Best-effort cancel (the active OpenSearch bulk batch will complete; the job transitions to `cancelled` once the worker thread returns) |

### Index and document admin

| Endpoint | Purpose |
|---|---|
| `GET /admin/index/stats` | Doc count, store size, primaries |
| `GET /admin/index/mapping` | Current index mapping |
| `POST /admin/index/refresh` | `_refresh` the index (useful right after ingest in tests) |
| `DELETE /admin/index?confirm=true` | Drop the index (`?confirm=true` is required) |
| `GET /admin/docs/{idno}` | Show all documents for an `idno` (helps verify ingest of a specific record) |
| `DELETE /admin/docs/{idno}` | `delete_by_query` for that `idno` |
| `POST /admin/embeddings/encode` `{texts, as_query?}` | Return vectors using the same `EmbeddingService` used by `/search`; useful for diagnosing dimension/prompt mismatches |
| `GET /admin/ml/pipeline` | Show expected and currently installed `text_embedding` ingest pipeline (when `embedding_backend=opensearch_ml`) |

### Quickstart (curl)

```bash
# (optional) gate admin endpoints
export NADA_ADMIN_API_KEY=dev-secret
H=(-H "X-NADA-Admin-Key: $NADA_ADMIN_API_KEY")

# Create the index
curl -s "${H[@]}" -X POST http://localhost:8020/admin/index \
  -H 'Content-Type: application/json' -d '{"recreate": false}'
# -> 202 {"id": "...", "kind": "create_index", "status": "pending", ...}

# Trigger catalog ingest (returns immediately)
curl -s "${H[@]}" -X POST http://localhost:8020/admin/ingest/from-catalog \
  -H 'Content-Type: application/json' \
  -d '{"catalog_type": "timeseries", "limit": 50}'
# -> 202 {"id": "abc...", ...}

# Re-trigger while running -> 409 with the SAME job id
curl -s "${H[@]}" -X POST http://localhost:8020/admin/ingest/from-catalog \
  -H 'Content-Type: application/json' -d '{"catalog_type": "timeseries"}'
# -> 409 {"detail": "a job with this key is already running", "job": {...}}

# Poll
curl -s http://localhost:8020/jobs/abc...
curl -s 'http://localhost:8020/jobs?status=running'

# Index stats
curl -s "${H[@]}" http://localhost:8020/admin/index/stats
```

## Demos

**CLI (live catalog + OpenSearch):**

```bash
export NADA_EMBEDDING_MODEL_ID=avsolatorio/GIST-small-Embedding-v0
export NADA_QUERY_PROMPT_NAME=
uv run python -m nada_ai.demo_integration --max_items=5
```

**Browser:** start the API (`uvicorn nada_ai.app.main:app --reload --port 8020`) and open **`GET /demo`** for a minimal UI that posts to **`POST /search`**.

## Tests

Fast tests (no cluster):

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # if a broken global pytest plugin loads
uv run pytest tests -q
# Coverage (load pytest-cov explicitly when autoload is disabled):
uv run pytest tests --cov=nada_ai --cov-report=term-missing -p pytest_cov
```

Integration tests against a live OpenSearch cluster are optional (`@pytest.mark.integration`) and can be run in CI or manually when compose is up. With `NADA_INTEGRATION_OPENSEARCH=1` and network access to the catalog API, `tests/integration/test_index_from_catalog_live.py` runs a small `index_from_catalog` (`limit=2`) end-to-end.

## Layout

| Area | Role |
|------|------|
| `nada_ai.ingest` | Catalog-driven bulk indexing (`MetadataLoader`, CLI, shared `service.*_op` callables) |
| `nada_ai.search` | Backend-agnostic surface (expand with protocols) |
| `nada_ai.search.backend.opensearch` | Client, mappings, ML ingest pipeline, embeddings, query DSL |
| `nada_ai.app` | FastAPI service (search, health, admin/ingest, jobs) |
| `nada_ai.app.jobs` | In-memory single-flight job registry (`JobRegistry`) backing `/admin/*` and `/jobs/*` |

**Import rule:** application code should import **`ai4data.discovery.*` only** (see `tests/test_ai4data_import_boundary.py`). Do not import `ai4data.config` etc. from package code.

**Discovery processors:** `register_discovery_processors()` from `ai4data.discovery.wiring` is only needed if you use PDF embedding helpers (`embed_documents` / `get_doc_reps`). Typical MetadataLoader + catalog ingest does **not** require it.

## MCP

Not implemented in this phase; reserve `nada_ai/mcp/` or an optional `[project.optional-dependencies] mcp = [...]` when you add MCP servers.