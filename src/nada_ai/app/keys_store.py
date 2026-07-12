"""File-based store for per-caller API keys (role-scoped, revocable).

Mirrors the atomic-write pattern already used for the facets config
(``search/dynamic_filters.py``): a JSON file, replaced atomically via
``tempfile.mkstemp`` + ``Path.replace`` so concurrent writers never
observe a partial file.

Only a SHA-256 hash of each key is ever persisted — API keys are
high-entropy random tokens (``secrets.token_urlsafe(32)``), not
human-chosen passwords, so a plain cryptographic hash (no per-key salt,
no slow KDF) is the same tradeoff most API providers make for this kind
of credential.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from nada_ai.settings import Settings

_DEFAULT_KEYS_PATH = Path(__file__).resolve().parents[3] / "config" / "api_keys.json"


class Role(StrEnum):
    read = "read"
    write = "write"
    admin = "admin"


ROLE_RANK: dict[Role, int] = {Role.read: 0, Role.write: 1, Role.admin: 2}


@dataclass
class KeyRecord:
    id: str
    name: str
    role: Role
    key_hash: str
    key_prefix: str
    created_at: str
    revoked_at: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "key_prefix": self.key_prefix,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
        }


def _resolve_keys_path(settings: Settings | None) -> Path:
    if settings is not None and settings.api_keys_path:
        return Path(settings.api_keys_path)
    return _DEFAULT_KEYS_PATH


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _load_all(settings: Settings | None) -> list[KeyRecord]:
    path = _resolve_keys_path(settings)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[KeyRecord] = []
    for raw in data.get("keys", []):
        try:
            out.append(
                KeyRecord(
                    id=raw["id"],
                    name=raw["name"],
                    role=Role(raw["role"]),
                    key_hash=raw["key_hash"],
                    key_prefix=raw.get("key_prefix", ""),
                    created_at=raw["created_at"],
                    revoked_at=raw.get("revoked_at"),
                )
            )
        except (KeyError, ValueError):
            continue
    return out


def _save_all(records: list[KeyRecord], settings: Settings | None) -> None:
    path = _resolve_keys_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": [
            {
                "id": r.id,
                "name": r.name,
                "role": r.role.value,
                "key_hash": r.key_hash,
                "key_prefix": r.key_prefix,
                "created_at": r.created_at,
                "revoked_at": r.revoked_at,
            }
            for r in records
        ]
    }
    content = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        try:
            os.write(fd, content)
        finally:
            os.close(fd)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


async def has_any_active_keys(settings: Settings | None) -> bool:
    records = await asyncio.to_thread(_load_all, settings)
    return any(r.revoked_at is None for r in records)


async def verify_key(raw_key: str | None, settings: Settings | None) -> KeyRecord | None:
    """Return the matching active ``KeyRecord``, or ``None`` if invalid/revoked."""
    if not raw_key:
        return None
    presented_hash = _hash_key(raw_key)
    records = await asyncio.to_thread(_load_all, settings)
    for record in records:
        if record.revoked_at is None and hmac.compare_digest(record.key_hash, presented_hash):
            return record
    return None


async def list_keys(settings: Settings | None) -> list[KeyRecord]:
    return await asyncio.to_thread(_load_all, settings)


async def create_key(
    name: str,
    role: Role,
    settings: Settings | None,
    lock: asyncio.Lock,
) -> tuple[KeyRecord, str]:
    """Create and persist a new key. Returns ``(record, raw_key)`` — the raw
    value is only ever available here; only its hash is stored."""
    raw_key = f"nada_{secrets.token_urlsafe(32)}"
    record = KeyRecord(
        id=uuid.uuid4().hex,
        name=name,
        role=role,
        key_hash=_hash_key(raw_key),
        key_prefix=raw_key[:12] + "...",
        created_at=datetime.now(UTC).isoformat(),
        revoked_at=None,
    )
    async with lock:
        records = await asyncio.to_thread(_load_all, settings)
        records.append(record)
        await asyncio.to_thread(_save_all, records, settings)
    return record, raw_key


async def revoke_key(
    key_id: str,
    settings: Settings | None,
    lock: asyncio.Lock,
) -> KeyRecord | None:
    async with lock:
        records = await asyncio.to_thread(_load_all, settings)
        found: KeyRecord | None = None
        for r in records:
            if r.id == key_id:
                if r.revoked_at is None:
                    r.revoked_at = datetime.now(UTC).isoformat()
                found = r
                break
        if found is None:
            return None
        await asyncio.to_thread(_save_all, records, settings)
        return found
