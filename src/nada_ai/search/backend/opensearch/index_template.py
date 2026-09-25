"""Composable index templates + optional cluster hardening for OpenSearch.

If an index is auto-created (e.g. first bulk without an explicit ``indices.create``), a matching **composable index
template** still applies the right mappings. There is one template per index, each matching its index name exactly:
the chunk index (``index_name``) and the study index (``studies_index``).

Cluster setting ``action.auto_create_index`` can be tightened (requires manager-level
permissions); see :class:`nada_ai.settings.Settings`.
"""

from __future__ import annotations

import logging
from typing import Any

from nada_ai.search.backend.opensearch.mapping import (
    citations_index_body,
    index_body,
    studies_index_body,
    variables_index_body,
)
from nada_ai.settings import Settings

logger = logging.getLogger(__name__)


def _template_name(index: str) -> str:
    """Stable template name derived from an index name (cluster-wide unique)."""
    return f"nada-ai-{index.replace('/', '-')}-template"


def composable_index_template_name(settings: Settings) -> str:
    """Template name of the chunk index."""
    return _template_name(settings.index_name)


def studies_index_template_name(settings: Settings) -> str:
    """Template name of the study index."""
    return _template_name(settings.studies_index)


def variables_index_template_name(settings: Settings) -> str:
    """Template name of the variable index."""
    return _template_name(settings.variables_index)


def citations_index_template_name(settings: Settings) -> str:
    """Template name of the citation index."""
    return _template_name(settings.citations_index)


def composable_index_template_body(settings: Settings, embedding_dimension: int | None) -> dict[str, Any]:
    """Body for ``indices.put_index_template`` of the chunk index."""
    return {
        "index_patterns": [settings.index_name],
        "template": index_body(embedding_dimension),
        "priority": settings.opensearch_index_template_priority,
    }


def studies_index_template_body(settings: Settings) -> dict[str, Any]:
    """Body for ``indices.put_index_template`` of the study index."""
    return {
        "index_patterns": [settings.studies_index],
        "template": studies_index_body(),
        "priority": settings.opensearch_index_template_priority,
    }


def variables_index_template_body(settings: Settings) -> dict[str, Any]:
    """Body for ``indices.put_index_template`` of the variable index."""
    return {
        "index_patterns": [settings.variables_index],
        "template": variables_index_body(),
        "priority": settings.opensearch_index_template_priority,
    }


def citations_index_template_body(settings: Settings) -> dict[str, Any]:
    """Body for ``indices.put_index_template`` of the citation index."""
    return {
        "index_patterns": [settings.citations_index],
        "template": citations_index_body(),
        "priority": settings.opensearch_index_template_priority,
    }


def put_composable_index_template(client: Any, settings: Settings, embedding_dimension: int | None) -> dict[str, Any]:
    """Install or replace the composable index templates of every index (chunks, studies, variables, citations)."""
    installed: dict[str, Any] = {}
    for name, body in (
        (composable_index_template_name(settings), composable_index_template_body(settings, embedding_dimension)),
        (studies_index_template_name(settings), studies_index_template_body(settings)),
        (variables_index_template_name(settings), variables_index_template_body(settings)),
        (citations_index_template_name(settings), citations_index_template_body(settings)),
    ):
        client.indices.put_index_template(name=name, body=body)
        logger.info("Installed composable index template %s patterns=%s", name, body["index_patterns"])
        installed[name] = {"index_patterns": body["index_patterns"], "priority": body["priority"]}
    return {"templates": installed}


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
