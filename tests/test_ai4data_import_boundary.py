"""
``nada_ai`` must depend on Data Compass integration only via ``ai4data.discovery`` (submodules allowed).

Guards against accidental imports from ``ai4data.config``, ``ai4data.interfaces``, etc.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _nada_py_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "nada_ai"
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _violations_in_tree(tree: ast.AST, path: str) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] != "ai4data":
                    continue
                if len(parts) == 1:
                    bad.append(f"{path}:{node.lineno}: import ai4data (use ai4data.discovery only)")
                    continue
                if parts[1] != "discovery":
                    bad.append(f"{path}:{node.lineno}: import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if not node.module:
                continue
            parts = node.module.split(".")
            if parts[0] != "ai4data":
                continue
            if len(parts) == 1:
                bad.append(f"{path}:{node.lineno}: from ai4data import …")
                continue
            if parts[1] != "discovery":
                bad.append(f"{path}:{node.lineno}: from {node.module} import …")
    return bad


def test_nada_ai_only_imports_ai4data_discovery() -> None:
    all_bad: list[str] = []
    for path in _nada_py_files():
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        rel = path.relative_to(path.parents[2])
        all_bad.extend(_violations_in_tree(tree, str(rel)))
    assert not all_bad, "Disallowed ai4data imports:\n" + "\n".join(all_bad)
