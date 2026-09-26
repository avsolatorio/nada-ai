"""The Qdrant writer confirms each point's write to the progress tracker (a study is checkpointed as done only once
all of its points are in the collection -- see ``IngestProgressTracker.expect``)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

from nada_ai.ingest.qdrant_writer import QdrantIngestWriter
from nada_ai.settings import Settings


class _Embedding:
    def embedding_dimension(self) -> int:
        return 3


def _record(point_id: str, sid: int) -> tuple[str, list[float], dict[str, Any]]:
    return point_id, [0.1, 0.2, 0.3], {"page_content": "text", "metadata": {"idno": f"S{sid}", "sid": sid}}


def _run(records: list[Any], upsert: Any) -> MagicMock:
    settings = Settings(search_backend="qdrant", embedding_backend="local", qdrant_sparse_lexical=False)
    client = MagicMock()
    client.upsert.side_effect = upsert
    progress = MagicMock()
    with (
        patch("nada_ai.ingest.qdrant_writer._client", return_value=client),
        patch("nada_ai.ingest.qdrant_writer.iter_langdoc_records", return_value=iter(records)),
        patch.object(QdrantIngestWriter, "ensure_target"),
    ):
        QdrantIngestWriter(settings).run_bulk([("S1", "microdata")], embedding=_Embedding(), progress=progress)
    return progress


def test_every_written_point_is_confirmed() -> None:
    progress = _run([_record("p1", 1), _record("p2", 1)], upsert=None)
    assert progress.confirm.call_args_list == [call(1, True, None), call(1, True, None)]


def test_a_rejected_point_is_confirmed_as_failed() -> None:
    def upsert(collection_name: str, points: list[Any], wait: bool) -> None:
        if any(p.id == "p2" for p in points):
            raise RuntimeError("bad payload")

    progress = _run([_record("p1", 1), _record("p2", 2)], upsert=upsert)
    # the batch fails, then each point is retried alone: p1 lands, p2 does not
    assert progress.confirm.call_args_list == [call(1, True, None), call(2, False, "bad payload")]
