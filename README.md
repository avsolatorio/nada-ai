# nada-ai (`nada_ai`)

Python package for **NADA AI**: ingest [NADA / Data Compass–style](https://data-compass.ihsn.org/) metadata via **`ai4data.discovery`** and search in **OpenSearch** (keyword, k-NN vector, hybrid). **MCP** support is planned as an optional extension in this repo.

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

## Configuration (`NADA_*`)

Settings use the **`NADA_`** prefix (see `nada_ai.settings`). Common variables:

| Variable | Purpose |
|----------|---------|
| `NADA_OPENSEARCH_URL` | Cluster URL (default `http://localhost:9200`) |
| `NADA_INDEX_NAME` | Index name (default `nada-metadata`) |
| `NADA_EMBEDDING_BACKEND` | `local` (default) or `opensearch_ml` |
| `NADA_EMBEDDING_MODEL_ID` | Hugging Face model id for `local` |
| `NADA_QUERY_PROMPT_NAME` | Optional asymmetric query prompt; empty for symmetric models |
| `NADA_OPENSEARCH_AUTH_MODE` | `basic` or `aws_sigv4` |
| `NADA_AWS_REGION` | Required for SigV4 when not implicit from boto3 |

**Discovery caches** (standalone `ai4data`):

| Variable | Purpose |
|----------|---------|
| `AI4DATA_DISCOVERY_DATA_PATH` | Writable directory for discovery caches and bundled metadata helpers (set in Docker and CLI jobs) |

Optional: call `init_discovery_paths(Path(...))` from `ai4data.discovery.config` at startup if you prefer code over env.

## Search and OpenSearch

The FastAPI app exposes **`POST /search`** (keyword, vector, hybrid), **`GET /demo`**, and health routes. Search query DSL lives under **`nada_ai.search.backend.opensearch`**.

### Version compatibility

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

Integration tests against a live OpenSearch cluster are optional (`@pytest.mark.integration`) and can be run in CI or manually when compose is up.

## Layout

| Area | Role |
|------|------|
| `nada_ai.ingest` | Catalog-driven bulk indexing (`MetadataLoader`, CLI) |
| `nada_ai.search` | Backend-agnostic surface (expand with protocols) |
| `nada_ai.search.backend.opensearch` | Client, mappings, ML ingest pipeline, embeddings, query DSL |
| `nada_ai.app` | FastAPI service |

**Import rule:** application code should import **`ai4data.discovery.*` only** (see `tests/test_ai4data_import_boundary.py`). Do not import `ai4data.config` etc. from package code.

**Discovery processors:** `register_discovery_processors()` from `ai4data.discovery.wiring` is only needed if you use PDF embedding helpers (`embed_documents` / `get_doc_reps`). Typical MetadataLoader + catalog ingest does **not** require it.

## MCP

Not implemented in this phase; reserve `nada_ai/mcp/` or an optional `[project.optional-dependencies] mcp = [...]` when you add MCP servers.
