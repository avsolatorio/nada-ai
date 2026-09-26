# nada-ai (`nada_ai`)

Python package for **NADA AI**: ingest [NADA / Data Compass–style](https://data-compass.ihsn.org/) metadata via **`ai4data.discovery`**, search it (keyword, k-NN vector, hybrid) over **OpenSearch or Qdrant**, and expose catalog search + timeseries analytics to LLM agents via an **MCP server** (17 tools, 15 interactive apps, resources, prompts).

**New here? Read the [developer guide](docs/GUIDE.md)** — architecture, semantic search, ingestion, the MCP server, NADA connectivity, every config variable, deployment, and observability, all in one place. This README only covers install and a pointer to the guides; it does not duplicate command sequences that could drift out of sync with them.

## Requirements

- Python **3.11+**
- [**uv**](https://docs.astral.sh/uv/) recommended
- **`ai4data[discovery]`** is pinned to a git revision in **`[tool.uv.sources]`** for reproducible installs and support for the catalog types used by this package. Update the revision in `pyproject.toml` and run **`uv lock`** to upgrade it. To develop against a local checkout, temporarily override the source with a path dependency.
- **Docker** (for a local Qdrant or OpenSearch instance) — on macOS with Docker Desktop, the `docker` CLI lives at `~/.docker/bin/docker` and is only on `PATH` in a **login shell**; a plain non-login shell (some editor/CI task runners) may need `export PATH="$HOME/.docker/bin:$PATH"` first.

## Install

From this directory:

```bash
cd nada-ai
uv sync --all-groups
# Local SentenceTransformer embeddings (default search/ingest path):
uv sync --extra local
```

For **Qdrant** vector search (the default backend — recommended local stack):

```bash
uv sync --extra local --extra qdrant
```

For **Amazon OpenSearch / IAM SigV4**:

```bash
uv sync --extra aws
```

Then **[jump to the developer guide's Quickstart](docs/GUIDE.md#quickstart)** for `.env` setup, bringing up the stack, and your first ingest + search.

## Guides

| Guide | Description |
|-------|-------------|
| **[Developer guide](docs/GUIDE.md)** | Full reference: architecture, semantic search, ingestion, the MCP server (tools/apps/resources/prompts), NADA connectivity, every config variable, deployment, observability |
| **[Qdrant pipeline guide](docs/qdrant-pipeline-guide.md)** | End-to-end catalog ingest and search with Qdrant — **host ingest** and **full Docker** setups, metadata-extract catalog, verification, troubleshooting |
| [Dynamic filters](docs/dynamic-filters.md) | Sync catalog filters from a NADA instance's metadata-extract API into the index and search by facet keys |

## Configuration

Settings use the **`NADA_`** prefix (see `nada_ai.settings`) for nada-ai's own config, and **`AI4DATA_`** for the `ai4data.discovery` package's catalog/credential config — both documented together in **[`.env.example`](.env.example)**, the authoritative, fully-commented list of every variable. Copy it to `.env` and edit:

```bash
cp .env.example .env
```

The two values you'll actually need to fill in for a non-default NADA instance are `AI4DATA_METADATA_CATALOG_URL` and `AI4DATA_METADATA_CATALOG_X_API_KEY` — see [Connecting to NADA](docs/GUIDE.md#connecting-to-nada) and the [full configuration reference](docs/GUIDE.md#configuration-reference) in the developer guide.

## Tests

```bash
uv run pytest -q
```

Integration tests against a live backend/catalog are gated behind env flags (`@pytest.mark.integration`) and skip cleanly without them — see [Testing](docs/GUIDE.md#testing) in the developer guide for the full list and how to run each one.
