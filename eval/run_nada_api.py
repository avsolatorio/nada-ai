"""Score a NADA installation's catalog search against the golden queries, whatever engine is behind it.

Calls NADA's public catalog API (``GET /api/catalog``), so it measures what a user gets: NADA's own search driver
(database, Qdrant, Qdrant + database, OpenSearch) with its filters, counts and hydration. Run it once per
configuration, then compare the saved runs:

    uv run python eval/run_nada_api.py --base http://localhost/nada-semantic/index.php --label db --json db.json
    uv run python eval/run_nada_api.py --base ... --label qdrant_db --json qdrant_db.json
    uv run python eval/run_nada_api.py --compare db.json qdrant_db.json

Relevance comes from ``golden_queries.json``. Only the first page of ``--top`` studies is fetched; ``found`` (the
size of the whole result) is what the "median size" column reports, while the precision of the whole result is
measured over those first ``--top`` studies.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from metrics import report, score_query, summarize  # noqa: E402

GOLDEN = Path(__file__).with_name("golden_queries.json")

#: golden filter key -> NADA catalog API parameter
FILTER_PARAMS = {"countries": "country"}


def catalog_search(base: str, query: dict[str, Any], top: int) -> tuple[list[int], int]:
    """``(ranked study ids of the first page, found)`` for one golden query."""
    params: dict[str, str] = {"sk": query["query"], "ps": str(top)}
    for key, values in query["filters"].items():
        if key not in FILTER_PARAMS:
            raise SystemExit(f"golden filter {key!r} has no NADA API parameter; add it to FILTER_PARAMS")
        params[FILTER_PARAMS[key]] = ",".join(str(v) for v in values)
    url = f"{base.rstrip('/')}/api/catalog?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        body = json.load(response)
    if body.get("status") != "success":
        raise SystemExit(f"{query['id']}: {body.get('errors') or body.get('message')}")
    result = body["result"]
    return [int(row["id"]) for row in result["rows"]], int(result["found"])


def run(base: str, top: int) -> list[dict[str, Any]]:
    rows = []
    for query in json.loads(GOLDEN.read_text())["queries"]:
        sids, found = catalog_search(base, query, top)
        metrics = score_query(query, sids)
        metrics["returned"] = found  # the size of the whole result, not of the page fetched
        if query["expect"] == "empty":
            metrics["pass"] = found == 0
        rows.append({"id": query["id"], "category": query["category"], "sids": sids, "found": found, "m": metrics})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="NADA base URL, e.g. http://localhost/nada-semantic/index.php")
    parser.add_argument("--label", default="run", help="name of this configuration in the report")
    parser.add_argument("--top", type=int, default=100, help="studies fetched per query (first page size)")
    parser.add_argument("--json", type=Path, help="save this run here")
    parser.add_argument("--compare", nargs="+", type=Path, metavar="RUN.json", help="compare saved runs and exit")
    args = parser.parse_args()

    if args.compare:
        runs = [json.loads(path.read_text()) for path in args.compare]
        print(report({r["label"]: summarize(r["queries"]) for r in runs}, full=False))
        return 0
    if not args.base:
        parser.error("--base is required unless --compare is used")

    rows = run(args.base, args.top)
    if args.json:
        args.json.write_text(json.dumps({"label": args.label, "queries": rows}, indent=1))
    print(report({args.label: summarize(rows)}, full=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
