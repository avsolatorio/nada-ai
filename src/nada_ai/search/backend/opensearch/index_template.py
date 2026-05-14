"""Composable index template + optional cluster hardening for OpenSearch.

If the index is auto-created (e.g. first bulk without an explicit ``indices.create``),
a matching **composable index template** still applies ``knn_vector`` + facet mappings.

Cluster setting ``action.auto_create_index`` can be tightened (requires manager-level
permissions); see :class:`nada_ai.settings.Settings`.
"""

from __future__ import annotations

import logging
from typing import Any

from nada_ai.search.backend.opensearch.mapping import index_body
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def composable_index_template_name(settings: Settings) -> str:
    """Stable template name derived from ``index_name`` (cluster-wide unique)."""
    safe = settings.index_name.replace("/", "-")
    return f"nada-ai-{safe}-template"


def composable_index_template_body(settings: Settings, embedding_dimension: int) -> dict[str, Any]:
    """Body for ``indices.put_index_template`` (OpenSearch composable templates)."""
    base = settings.index_name
    patterns = [base, f"{base}-*"]
    return {
        "index_patterns": patterns,
        "template": index_body(embedding_dimension),
        "priority": settings.opensearch_index_template_priority,
    }


def put_composable_index_template(client: Any, settings: Settings, embedding_dimension: int) -> dict[str, Any]:
    """Install or replace the composable index template for ``index_name`` (+ suffix pattern)."""
    name = composable_index_template_name(settings)
    body = composable_index_template_body(settings, embedding_dimension)
    client.indices.put_index_template(name=name, body=body)
    logger.info("Installed composable index template %s patterns=%s", name, body["index_patterns"])
    return {"template": name, "index_patterns": body["index_patterns"], "priority": body["priority"]}


def _normalize_auto_create_index(value: str) -> str | bool:
    """Map common env strings to JSON types accepted by OpenSearch."""
    s = value.strip()
    sl = s.lower()
    if sl == "false":
        return False
    if sl == "true":
        return True
    return s


def put_cluster_auto_create_index(client: Any, value: str) -> dict[str, Any]:
    """Persist ``cluster.routing.allocation.*`` sibling: ``action.auto_create_index``.

    ``value`` examples:

    - ``\"false\"`` — disable all automatic index creation (strict).
    - ``\"true\"`` — allow all (default-like).
    - ``\"+nada-metadata*,-*\"`` — allowlist style (OpenSearch/Elasticsearch pattern strings).
    """
    coerced = _normalize_auto_create_index(value)
    body: dict[str, Any] = {"persistent": {"action.auto_create_index": coerced}}
    resp = client.cluster.put_settings(body=body)
    out: dict[str, Any] = {
        "acknowledged": bool(resp.get("acknowledged", True)),
        "action.auto_create_index": coerced,
    }
    logger.info("Cluster persistent action.auto_create_index set to %r", coerced)
    return out
