# Local Docker setup — start & index (Qdrant)

> This page covers the **Qdrant** stack. For **OpenSearch**, the default backend, see the
> [OpenSearch setup guide](./opensearch-setup-guide.md#4-start-the-local-docker-stack).

Day-to-day commands for the **nada-ai** stack already running locally via Docker Compose (Qdrant + FastAPI, both containerized). For first-time setup, TLS/corporate-proxy issues, or host-side ingest instead of Docker, see the full [Qdrant pipeline guide](./qdrant-pipeline-guide.md).

**This host:**

| Item | Value |
|------|-------|
| Compose file | `docker-compose.qdrant.yml` |
| API container | `nada-ai-api-qdrant-dev` — port `8020` |
| Qdrant container | `nada-ai-qdrant-dev` — ports `6333` (HTTP), `6334` (gRPC) |
| Collection | `nada-metadata` |
| Admin API key | Not set (`NADA_ADMIN_API_KEY` unset) — admin/ingest routes are open, no `X-NADA-Admin-Key` header needed |

---

## Populating `.env`

`docker-compose.qdrant.yml` loads `.env` into both containers via `env_file` (the API container's own `environment:` block then overrides a couple of keys — see the table below). Start from the example:

```bash
cp .env.example .env
mkdir -p data/nada-discovery
```

**Required for this local Qdrant setup:**

| Variable | Value on this host | Notes |
|----------|---------------------|-------|
| `NADA_SEARCH_BACKEND` | `qdrant` | |
| `NADA_QDRANT_URL` | `http://localhost:6333` | Only read for host-side CLI use; inside the API container, compose's `environment:` block overrides this to `http://qdrant-nada:6333` regardless of what's in `.env` |
| `NADA_INDEX_NAME` | `nada-metadata` | |
| `NADA_EMBEDDING_BACKEND` | `local` | |
| `NADA_EMBEDDING_MODEL_ID` | `microsoft/harrier-oss-v1-270m` | |
| `AI4DATA_DISCOVERY_DATA_PATH` | `./data/nada-discovery` | Create the directory first (above) |
| `AI4DATA_METADATA_CATALOG_URL` | your NADA instance's `index.php` URL | If NADA runs on the **same Mac** (e.g. a local Apache/MAMP vhost), use `http://host.docker.internal/...`, **not** `http://localhost/...` — inside the API container, `localhost` resolves to the container itself, not your Mac, and ingest jobs fail instantly with `ConnectError` |
| `AI4DATA_METADATA_CATALOG_X_API_KEY` | a NADA **admin** API key | Required because extract mode (next row) hits an admin-only endpoint — see [below](#where-to-get-the-catalog-api-key) |

**Optional:**

| Variable | When to set it |
|----------|-----------------|
| `AI4DATA_METADATA_CATALOG_EXTRACT_PATH` | Leave as `api/admin/search-metadata-extract` for fast bulk ingest (paginates once instead of one request per idno). Comment it out if you don't have an admin key — ingest then falls back to the slower classic per-idno flow, which works anonymously |
| `NADA_ADMIN_API_KEY` | Locks the API's own `/admin/*` and `/jobs` routes behind an `X-NADA-Admin-Key` header. Unset = those routes are fully open — fine for local dev on `localhost`, not for anything reachable over a network |
| `AI4DATA_METADATA_CATALOG_COOKIES` / `AI4DATA_METADATA_CATALOG_AUTH_BEARER` | Alternate/additional catalog auth some NADA instances require instead of, or alongside, the API key |
| `NADA_EMBEDDING_DEVICE` | `mps` for host-side ingest on Apple Silicon; compose always forces `cpu` inside the container |

### Where to get the catalog API key

`AI4DATA_METADATA_CATALOG_X_API_KEY` is generated **from the target NADA instance itself** (its admin panel, or whoever administers that instance) — it's an *outbound* credential (nada-ai authenticating to the catalog). It is a different trust boundary from `NADA_ADMIN_API_KEY`, which is *inbound* (callers authenticating to nada-ai) — never reuse the same secret for both; a leak of one would otherwise grant both catalog access and full control of this search index.

After editing `.env`, recreate the API container so it picks up the change (Qdrant itself doesn't need restarting for a catalog-credential edit):

```bash
docker compose -f docker-compose.qdrant.yml up -d --force-recreate nada-ai-api
```

---

## How to start

```bash
cd nada-ai
docker compose -f docker-compose.qdrant.yml up -d
```

The plain compose file `COPY`s `src/` into the image at build time (see `Dockerfile`) — it does **not** bind-mount source. That means **any** code change, including a `git pull`/branch switch, needs a rebuild to actually take effect; the container will keep serving the old code (silently — no error, just missing routes/fixes/404s) otherwise:

```bash
docker compose -f docker-compose.qdrant.yml up --build -d
```

To skip rebuilding on every edit during active development, use the dev overlay instead, which bind-mounts `src/` and enables uvicorn `--reload`:

```bash
docker compose -f docker-compose.qdrant.yml -f docker-compose.dev.yml up -d
```

**Verify it's up:**

```bash
docker compose -f docker-compose.qdrant.yml ps

curl -s http://localhost:6333/                          # Qdrant
curl -s http://localhost:8020/health                     # API
curl -s http://localhost:8020/health/embeddings           # embedding backend
```

Warm the embedding model so the first search isn't slow:

```bash
curl -s -X POST http://localhost:8020/health/embeddings/warmup
```

Browser demo UI: **http://localhost:8020/demo**

**Logs:**

```bash
docker logs nada-ai-api-qdrant-dev 2>&1 | tail -50
docker logs nada-ai-qdrant-dev 2>&1 | tail -50
```

**Stop:**

```bash
docker compose -f docker-compose.qdrant.yml down       # keep Qdrant data volume
docker compose -f docker-compose.qdrant.yml down -v    # also wipe Qdrant data
```

---

## How to index

Two ways to run ingest: `docker exec` into the API container (foreground, blocks until done) or the HTTP admin API (background job, non-blocking — preferred for full-catalog runs).

### Option 1 — HTTP admin API (background jobs, recommended)

Index every catalog type (`document`, `timeseries`, `survey`, `geospatial`) in one call:

```bash
curl -X POST http://localhost:8020/admin/ingest/from-catalog/all \
  -H "Content-Type: application/json" \
  -d '{}'
```

Or a single type:

```bash
curl -X POST http://localhost:8020/admin/ingest/from-catalog \
  -H "Content-Type: application/json" \
  -d '{"catalog_type": "document", "ps": 100}'
```

Optional body fields: `limit` (smoke test, e.g. `10`), `force` (re-fetch/re-embed cached ids), `recreate_index` (drops and recreates `nada-metadata` first — do not use unless wiping).

Both return `202` with a job id per catalog type. Re-posting while a type is still running is safe — it reports `already_running: true` instead of starting a duplicate.

**Check status:**

```bash
# All jobs (optionally ?status=running)
curl -s http://localhost:8020/jobs | python3 -m json.tool

# One job by id (from the POST response above)
curl -s http://localhost:8020/jobs/<job_id> | python3 -m json.tool

# Point count in Qdrant should keep climbing while a job runs
curl -s http://localhost:6333/collections/nada-metadata | python3 -m json.tool
```

If every job's `status` flips straight to `failed` with `"error": "ConnectError"` within a second of submitting, `AI4DATA_METADATA_CATALOG_URL` is unreachable from inside the container — almost always because it's set to `http://localhost/...` pointing at a catalog that actually runs on the host Mac. Use `host.docker.internal` instead (see [Populating .env](#populating-env)), then `docker compose -f docker-compose.qdrant.yml up -d --force-recreate nada-ai-api` and resubmit.

> If `NADA_ADMIN_API_KEY` gets set later, add `-H "X-NADA-Admin-Key: $NADA_ADMIN_API_KEY"` to every admin/ingest call above.

### Option 2 — `docker exec` (foreground CLI)

Create the collection (first time only, or after a wipe):

```bash
docker exec nada-ai-api-qdrant-dev \
  python -m nada_ai.ingest.cli create_index
```

Smoke test (10 records):

```bash
docker exec nada-ai-api-qdrant-dev \
  python -m nada_ai.ingest.cli index_from_catalog \
  --catalog_type=document --limit=10
```

Full ingest, one type (add `-it` to see progress live; can take a while):

```bash
docker exec -it nada-ai-api-qdrant-dev \
  python -m nada_ai.ingest.cli index_from_catalog \
  --catalog_type=document --ps=100
```

Every type:

```bash
for t in document timeseries survey geospatial; do
  docker exec -it nada-ai-api-qdrant-dev \
    python -m nada_ai.ingest.cli index_from_catalog \
    --catalog_type="$t" --ps=100
done
```

`--catalog_type` aliases: `indicator` → `timeseries`, `microdata` → `survey`. Metadata cache is written to `./data/nada-discovery` on the host (bind-mounted).

### Verify ingest worked

```bash
curl -s http://localhost:6333/collections/nada-metadata | python3 -m json.tool   # points_count > 0

curl -s -X POST http://localhost:8020/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"poverty","mode":"hybrid","size":5}' | python3 -m json.tool
```

---

## Related docs

- [Qdrant pipeline guide](./qdrant-pipeline-guide.md) — full setup, TLS/corporate-proxy notes, troubleshooting table
- [Architecture](./architecture.md)
- [README](../README.md) — full admin/ingest HTTP API reference
