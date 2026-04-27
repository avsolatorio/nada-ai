from __future__ import annotations

import json
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

EmbeddingBackend = Literal["local", "opensearch_ml"]


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
