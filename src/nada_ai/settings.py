from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

EmbeddingBackend = Literal["local", "opensearch_ml"]
SearchBackendKind = Literal["opensearch", "qdrant"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NADA_", case_sensitive=False, extra="ignore")

    opensearch_url: str = Field(default="http://localhost:9200")
    opensearch_user: str | None = Field(default=None)
    opensearch_password: str | None = Field(default=None)
    opensearch_ca_certs: str | None = Field(default=None)
    opensearch_verify_certs: bool = Field(default=True)
    opensearch_auth_mode: Literal["basic", "aws_sigv4"] = Field(
        default="basic",
        description="basic: user/password; aws_sigv4: IAM (install optional extra `aws` for boto3).",
    )
    aws_region: str | None = Field(default=None, description="AWS region for SigV4; defaults from boto3 session")
    aws_service: str = Field(
        default="es", description="AWS SigV4 service name (es for OpenSearch domain; aoss for Serverless)"
    )
    aws_profile: str | None = Field(default=None, description="Optional boto3 profile name")

    index_name: str = Field(default="nada-metadata")

    #: If True, ``PUT _index_template`` before index create / bulk ingest so auto-created indices inherit ``knn_vector`` mapping.
    opensearch_put_composable_index_template: bool = Field(default=True)
    #: Composable template ``priority`` (higher wins when multiple templates match).
    opensearch_index_template_priority: int = Field(default=500, ge=0, le=2000)

    #: If set, persist cluster ``action.auto_create_index`` (requires cluster-manager permissions). Examples: ``false`` to disable all auto-create; ``+nada-metadata*,-*`` for allowlist. Leave unset to not touch cluster settings.
    opensearch_cluster_auto_create_index: str | None = Field(default=None)

    #: Active vector / keyword search engine.
    search_backend: SearchBackendKind = Field(default="qdrant")

    #: Qdrant HTTP API (``host:port`` or full URL per ``qdrant-client``).
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_prefer_grpc: bool = Field(default=False)
    #: Defaults to ``index_name`` when unset for a single knob across engines.
    qdrant_collection_name: str | None = Field(default=None)

    #: ``local``: SentenceTransformers embed + k-NN queries. ``opensearch_ml``: ingest pipeline ``text_embedding`` + ``neural`` queries (no local model).
    embedding_backend: EmbeddingBackend = Field(default="local")

    embedding_model_id: str = Field(default="microsoft/harrier-oss-v1-270m")
    query_prompt_name: str | None = Field(default="web_search_query")
    #: Literal prefix for asymmetric query encoding (``SentenceTransformer.encode(..., prompt=...)``).
    #: If unset, ``query_prompt_name`` is used when set; otherwise encoding is symmetric.
    query_prompt: str | None = Field(
        default=None,
        description=(
            "Optional. E.g. 'Instruct: Retrieve semantically similar text\\nQuery: '. "
            "When set, overrides query_prompt_name for encode_query."
        ),
    )

    embedding_model_kwargs_json: str | None = Field(default='{"dtype": "auto"}')
    embedding_device: str | None = Field(default=None)
    embedding_batch_size: int = Field(default=32)

    #: Deployed ML Commons model id (``_plugins/_ml/models``). Required when ``embedding_backend=opensearch_ml``.
    opensearch_ml_model_id: str | None = Field(default=None)
    #: Vector length produced by that model; must match ``knn_vector`` mapping. Common: 384, 768, 1024.
    opensearch_ml_embedding_dimension: int | None = Field(default=None)
    #: Ingest pipeline with a ``text_embedding`` processor (``page_content`` → ``embedding``).
    opensearch_ml_ingest_pipeline_name: str = Field(default="nada-text-embedding")
    #: If True, skip ``PUT _ingest/pipeline`` (pipeline already exists with same definition).
    opensearch_ml_skip_ingest_pipeline_setup: bool = Field(default=False)

    hybrid_keyword_boost: float = Field(default=0.3)
    hybrid_vector_boost: float = Field(default=0.7)

    #: Qdrant only: ``query_points`` / ``query_points_groups`` ``score_threshold`` (min similarity score; cosine ⇒ higher = closer). Per-request JSON overrides this.
    qdrant_vector_score_threshold: float | None = Field(default=None)
    #: Qdrant only: when counting neighbors above ``qdrant_vector_score_threshold``, stop after this many points (cost cap).
    qdrant_vector_count_scan_cap: int = Field(default=100_000, ge=1, le=10_000_000)

    #: Qdrant: FastEmbed BM25 sparse vectors for ranked keyword/hybrid lexical leg (requires ``fastembed`` from the ``qdrant`` extra; recreate collection if enabling on an existing dense-only collection).
    qdrant_sparse_lexical: bool = Field(default=True)
    #: Named sparse vector in Qdrant; must match collection ``sparse_vectors_config`` and ingest upserts.
    qdrant_sparse_vector_name: str = Field(default="bm25")
    #: ``SparseTextEmbedding`` model id (e.g. ``Qdrant/bm25``); install ``uv sync --extra qdrant`` for ``fastembed``.
    qdrant_sparse_model_id: str = Field(default="Qdrant/bm25")
    #: When ``collapse_field`` is set on hybrid Qdrant search, multiply lexical/dense prefetch by this for post-RRF grouping.
    qdrant_hybrid_collapse_prefetch_multiplier: float = Field(default=4.0, ge=1.0, le=50.0)

    #: Optional path to JSON registry of facetable dynamic filter keys (see ``config/dynamic_filter_facets.json``).
    dynamic_filter_facets_path: str | None = Field(default=None)

    #: Maximum number of indexing jobs that may run embedding compute simultaneously.
    #: Set to 1 for GPU deployments (single-threaded model inference avoids OOM).
    #: Raise to 2–4 for CPU-only deployments with plentiful cores.
    max_concurrent_ingest_jobs: int = Field(default=1, ge=1, le=16)

    #: When true (default), content ingest (index/index_from_catalog) also
    #: fetches and bakes in each idno's NADA filters/facets at write time —
    #: see ingest.pipeline._fetch_filter_payload — instead of requiring a
    #: separate filters-sync pass afterward. Disable for a pure content-only
    #: ingest (e.g. to shave the extra per-idno metadata-extract round trip
    #: when the deployment doesn't use facets at all).
    sync_filters_during_ingest: bool = Field(default=True)

    #: NADA search-metadata-extract API base URL (no trailing slash). This is
    #: instance-specific — every NADA deployment (IHSN's or anyone else's) has
    #: its own metadata-extract host, so there is no built-in default here.
    #: Defaults to ``{AI4DATA_METADATA_CATALOG_URL}/{AI4DATA_METADATA_CATALOG_EXTRACT_PATH}``
    #: when unset — only set this if your extract API lives at a different
    #: host/path than that derivation produces (e.g. a nonstandard reverse
    #: proxy route). Credentials are NOT configured here — see
    #: ``nada_ai.nada.admin_auth``: both this and the search-index endpoints
    #: below always use ``AI4DATA_METADATA_CATALOG_X_API_KEY``/``_AUTH_BEARER``/
    #: ``_COOKIES``, since NADA's own docs require the same admin account for
    #: both admin surfaces — there is no separate credential to set here.
    metadata_extract_base_url: str | None = Field(default=None)

    #: NADA admin API root (the OpenAPI ``servers`` base, e.g.
    #: ``https://your-nada-instance/index.php/api``) for the ``search-index``
    #: change-queue endpoints. Defaults to ``{AI4DATA_METADATA_CATALOG_URL}/api``
    #: when unset — same rare-override rationale as ``metadata_extract_base_url``
    #: above, and the same shared ``AI4DATA_METADATA_CATALOG_*`` credentials.
    search_index_base_url: str | None = Field(default=None)

    #: Run the search-index queue reconciliation as an in-process periodic
    #: loop inside the FastAPI app (see app/reconcile_scheduler.py), instead
    #: of only via the manual `reconcile_search_index` CLI command. Off by
    #: default — enabling it requires an admin credential for the target NADA
    #: instance (see metadata_extract_* above) and NADA's own search-index
    #: tracking configured for this deployment (search_provider set,
    #: tracking_enabled true — check with `search_index_status` first).
    reconcile_search_index_enabled: bool = Field(default=False)
    #: Seconds between reconciliation polls when enabled.
    reconcile_search_index_interval_seconds: int = Field(default=300, ge=30, le=3600)
    #: Max queue items to submit as jobs per poll (mirrors list_queue's own cap).
    reconcile_search_index_batch_limit: int = Field(default=50, ge=1, le=100)

    #: Override path to the API keys store (default ``config/api_keys.json``).
    #: Contains only key hashes/prefixes, never raw key values — still keep out of version control.
    api_keys_path: str | None = Field(default=None)
    #: Override path to the admin audit log (default ``config/audit.log``, JSONL, append-only).
    audit_log_path: str | None = Field(default=None)
    #: Per-minute request cap (per caller: admin key if presented, else client IP) for the public
    #: /search, /recommendations, /search/explain, and PDF preview endpoints. 0 disables.
    rate_limit_search_per_minute: int = Field(default=120, ge=0)

    #: Root logger format for the FastAPI process. ``json`` for log-aggregator environments.
    log_format: Literal["text", "json"] = Field(default="text")
    #: Root logger level for the FastAPI process (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    log_level: str = Field(default="INFO")

    @field_validator("opensearch_cluster_auto_create_index", mode="before")
    @classmethod
    def empty_auto_create_to_none(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v
        return str(v)

    @field_validator("query_prompt_name", "query_prompt", mode="before")
    @classmethod
    def empty_prompt_strings_to_none(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v
        return str(v)

    @property
    def embedding_model_kwargs(self) -> dict[str, Any]:
        if not self.embedding_model_kwargs_json:
            return {}
        try:
            return json.loads(self.embedding_model_kwargs_json)
        except json.JSONDecodeError:
            return {}

    def describe_query_encoding(self) -> dict[str, Any]:
        """How ``EmbeddingService.encode_query`` will call the model (local backend only)."""
        if self.query_prompt:
            return {
                "active": "literal_prompt",
                "prompt": self.query_prompt,
                "prompt_name_configured": self.query_prompt_name,
            }
        if self.query_prompt_name:
            return {"active": "prompt_name", "prompt_name": self.query_prompt_name}
        return {"active": "symmetric"}

    @property
    def qdrant_collection(self) -> str:
        return self.qdrant_collection_name or self.index_name

    @model_validator(mode="after")
    def validate_search_backend(self) -> Settings:
        if self.search_backend == "qdrant" and self.embedding_backend == "opensearch_ml":
            raise ValueError(
                "embedding_backend opensearch_ml is only valid with search_backend=opensearch (OpenSearch ML Commons)."
            )
        return self

    @model_validator(mode="after")
    def validate_ml_backend(self) -> Settings:
        if self.embedding_backend == "opensearch_ml":
            if not self.opensearch_ml_model_id:
                raise ValueError(
                    "opensearch_ml_model_id (NADA_OPENSEARCH_ML_MODEL_ID) is required when embedding_backend is opensearch_ml"
                )
            if self.opensearch_ml_embedding_dimension is None:
                raise ValueError(
                    "opensearch_ml_embedding_dimension (NADA_OPENSEARCH_ML_EMBEDDING_DIMENSION) is required when "
                    "embedding_backend is opensearch_ml"
                )
        return self


class MCPServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NADA_MCP_", case_sensitive=False, extra="ignore")

    port: int = Field(default=8025, description="Port to bind the MCP server to.")
    transport: str = Field(default="http", description="Transport to use for the MCP server.")
    log_file: str | None = Field(default=None, description="Path to log file. If None, logs go to stderr/stdout.")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    env: str | None = Field(default=None, description="Deployment environment (e.g. local, dev, staging, prod).")
    server_name: str | None = Field(
        default=None,
        description="Display name for the MCP server (default: NADA MCP Server).",
    )
    tool_prefix: str = Field(
        default="nada",
        description="Prefix for MCP tool names, e.g. 'wdr' -> wdr_search_catalog (lowercase letters, digits, underscores).",
    )
    catalog_name: str = Field(
        default="NADA catalog",
        description="Human-readable catalog label inserted into default MCP tool descriptions.",
    )
    search_catalog_description: str | None = Field(
        default=None,
        description="Full override for the search_catalog MCP tool description shown to LLMs.",
    )
    get_metadata_description: str | None = Field(
        default=None,
        description="Full override for the get_metadata MCP tool description shown to LLMs.",
    )
    readiness_enabled: bool = Field(
        default=True,
        description="When false, GET /ready returns 200 with readiness_checks=disabled (no dependency probes).",
    )
    health_check_timeout: float = Field(
        default=5.0,
        description="Per-check timeout in seconds for GET /ready outbound probes.",
    )


def get_mcp_server_settings() -> MCPServerSettings:
    return MCPServerSettings()  # pyright: ignore[reportCallIssue]


def setup_mcp_logging(
    log_file: str | None = None,
    log_level: str = "INFO",
    env: str | None = None,
) -> None:
    """Configure logging to write to a file and/or console.

    Args:
        log_file: Path to log file. If None, logs only go to stderr.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        env: Deployment environment.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        log_path = Path(log_file)
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
