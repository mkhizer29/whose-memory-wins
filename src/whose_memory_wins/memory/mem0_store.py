"""Thin adapter over the Mem0 open-source ``Memory`` class (tested with mem0ai==2.2.1).

Verified Mem0 behaviour this adapter relies on (read from the installed source, see docs):
* ``add(messages, user_id=..., metadata=..., infer=False)`` stores each non-system message
  verbatim as one memory (no LLM call). Used for Track 1 (controlled insertion).
* ``add(..., infer=True)`` runs Mem0's LLM extraction. In 2.2.1 this pipeline is additive
  (it only reports ADD events) and shows the extraction LLM up to 10 previous session
  messages and up to 10 similar existing memories. Used for Track 2 (normal ingestion).
* ``search(query, filters={"user_id": ...}, top_k=..., threshold=...)`` returns scored results.
* ``update``/``delete``/``history`` exist and are covered by tests.

Because Mem0 does not promise stable IDs or complete event reports, every write is wrapped
in a before/after snapshot diff of the namespace, so creations, updates and deletions are
detected from the stored state itself rather than assumed from the return value.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import env  # noqa: F401  (sets MEM0_TELEMETRY / cache paths before mem0 import)

LIST_LIMIT = 1000


@dataclass
class ChangeSet:
    reported: list[dict[str, Any]]                  # what Mem0's add() returned
    created: list[dict[str, Any]] = field(default_factory=list)
    updated: list[dict[str, Any]] = field(default_factory=list)   # {"id", "old", "new"}
    deleted: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"reported": self.reported, "created": self.created, "updated": self.updated, "deleted": self.deleted}


def build_mem0_config(store_dir: Path, retrieval: dict, llm: dict | None) -> dict:
    cfg: dict[str, Any] = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "wmw", "path": str(store_dir / "qdrant"),
            "embedding_model_dims": retrieval["embedding_dims"], "on_disk": True}},
        "embedder": {"provider": retrieval["embedder_provider"], "config": {"model": retrieval["embedder_model"]}},
        "history_db_path": str(store_dir / "history.db"),
    }
    if llm is not None:
        cfg["llm"] = llm
    else:
        # Controlled insertion never calls the LLM; a placeholder config avoids Mem0's default
        # (which would try to construct an OpenAI client). It is never invoked with infer=False.
        cfg["llm"] = {"provider": "anthropic", "config": {"model": "unused-for-infer-false",
                                                          "api_key": "unused", "enable_sampling_parameters": False}}
    return cfg


class Mem0Store:
    def __init__(self, store_dir: Path, retrieval: dict, llm: dict | None = None):
        from mem0 import Memory  # imported lazily: heavy

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval = retrieval
        self.config = build_mem0_config(self.store_dir, retrieval, llm)
        self.memory = Memory.from_config(self.config)

    # ---- reads -------------------------------------------------------------------------
    def records(self, user_id: str) -> dict[str, dict[str, Any]]:
        res = self.memory.get_all(filters={"user_id": user_id}, top_k=LIST_LIMIT)["results"]
        return {r["id"]: r for r in res}

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self.memory.get(record_id)

    def search(self, user_id: str, query: str, top_k: int, threshold: float) -> list[dict[str, Any]]:
        res = self.memory.search(query, filters={"user_id": user_id}, top_k=top_k, threshold=threshold)["results"]
        return [{"rank": i + 1, "id": r["id"], "text": r["memory"], "score": r.get("score"),
                 "metadata": r.get("metadata") or {}} for i, r in enumerate(res)]

    def namespace_digest(self, user_id: str) -> str:
        recs = sorted((r["id"], r["memory"]) for r in self.records(user_id).values())
        return hashlib.sha256(json.dumps(recs).encode()).hexdigest()

    # ---- writes ------------------------------------------------------------------------
    def _diffed(self, user_id: str, fn) -> ChangeSet:
        before = self.records(user_id)
        reported = fn()
        after = self.records(user_id)
        cs = ChangeSet(reported=reported)
        for rid, rec in after.items():
            if rid not in before:
                cs.created.append({"id": rid, "text": rec["memory"]})
            elif rec["memory"] != before[rid]["memory"]:
                cs.updated.append({"id": rid, "old": before[rid]["memory"], "new": rec["memory"]})
        for rid, rec in before.items():
            if rid not in after:
                cs.deleted.append({"id": rid, "text": rec["memory"]})
        return cs

    def insert_verbatim(self, user_id: str, text: str, metadata: dict[str, Any]) -> tuple[str, ChangeSet]:
        """Controlled insertion via Mem0's documented ``infer=False`` path, verified by readback."""
        cs = self._diffed(user_id, lambda: self.memory.add(
            [{"role": "user", "content": text}], user_id=user_id, metadata=metadata, infer=False)["results"])
        if len(cs.created) != 1 or cs.updated or cs.deleted:
            raise RuntimeError(f"controlled insert produced unexpected changes: {cs.to_dict()}")
        rid = cs.created[0]["id"]
        stored = self.get(rid)
        if stored is None or stored["memory"] != text:
            raise RuntimeError(f"readback mismatch for {rid}: {stored!r}")
        return rid, cs

    def ingest(self, user_id: str, text: str, sender_name: str, metadata: dict[str, Any]) -> ChangeSet:
        """Normal Mem0 ingestion (LLM extraction)."""
        msg = {"role": "user", "content": text, "name": sender_name}
        return self._diffed(user_id, lambda: self.memory.add([msg], user_id=user_id, metadata=metadata,
                                                             infer=True)["results"])

    def update(self, user_id: str, record_id: str, text: str) -> ChangeSet:
        return self._diffed(user_id, lambda: [self.memory.update(record_id, text=text)])

    def delete(self, user_id: str, record_id: str) -> ChangeSet:
        return self._diffed(user_id, lambda: [self.memory.delete(record_id)])

    def history(self, record_id: str) -> list[dict[str, Any]]:
        return self.memory.history(record_id)

    def close(self) -> None:
        # Mem0 2.2.1's Memory.close() only closes the SQLite history DB; the local Qdrant
        # clients (memories + entity store) keep a file lock on Windows until closed.
        for obj in (getattr(self.memory, "vector_store", None), getattr(self.memory, "_entity_store", None)):
            client = getattr(obj, "client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
        try:
            self.memory.close()
        except Exception:  # pragma: no cover
            pass


def sanitize_name(name: str) -> str:
    """Mem0/OpenAI-style ``name`` fields allow only simple characters."""
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)[:64]


def dir_is_store(path: Path) -> bool:
    return (Path(path) / "qdrant").exists() and os.path.exists(Path(path) / "history.db")
