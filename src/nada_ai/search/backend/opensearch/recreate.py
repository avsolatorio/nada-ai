"""Drop an index and create it empty with the current mapping (sync client), for a mapping change an existing index
never takes: a strict index refuses every document carrying a field it lacks."""

from __future__ import annotations

from typing import Any

from opensearchpy.exceptions import NotFoundError, RequestError

from nada_ai.search.backend.opensearch.mapping import new_index_generation

#: Delete-and-create rounds before a recreate gives up (each further round follows a write that created the index in
#: between).
RECREATE_ATTEMPTS = 3


def recreate_index(client: Any, name: str, body: dict[str, Any]) -> str:
    """Drop ``name`` and create it from ``body`` stamped with a new generation; returns the generation.

    A write that lands between the delete and the create (a document indexed meanwhile) creates the index itself, so
    the create is refused as existing: the index is then deleted and created again.
    """
    generation = new_index_generation()
    body = {**body, "mappings": {**body["mappings"], "_meta": {"generation": generation}}}
    for attempt in range(RECREATE_ATTEMPTS):
        try:
            client.indices.delete(index=name)
        except NotFoundError:
            pass
        try:
            client.indices.create(index=name, body=body)
            return generation
        except RequestError as e:
            if e.error != "resource_already_exists_exception" or attempt == RECREATE_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")
