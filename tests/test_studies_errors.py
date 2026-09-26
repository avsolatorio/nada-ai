"""Which contract error a failed OpenSearch call becomes (``studies_errors.opensearch_error``)."""

from __future__ import annotations

from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import RequestError, TransportError

from nada_ai.app.studies_errors import opensearch_error
from nada_ai.app.studies_schemas import ERROR_HTTP_STATUS, ErrorCode


def test_a_rejected_query_is_query_rejected_not_an_outage() -> None:
    """A 400 means OpenSearch was up and refused what nada-ai built. A 5xx would make NADA count an outage."""
    error = opensearch_error("POST /x", RequestError(400, "search_phase_execution_exception", {"error": "..."}))
    assert error.code is ErrorCode.query_rejected
    assert ERROR_HTTP_STATUS[error.code] == 400
    assert error.details == {"engine_error": "search_phase_execution_exception"}


def test_no_connection_or_a_server_error_is_backend_unavailable() -> None:
    for e in (
        OpenSearchConnectionError("N/A", "refused", Exception("refused")),
        TransportError(503, "cluster_block_exception", {}),
    ):
        error = opensearch_error("POST /x", e)
        assert error.code is ErrorCode.backend_unavailable
        assert ERROR_HTTP_STATUS[error.code] == 503
