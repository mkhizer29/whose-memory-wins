"""Application-owned source register (SQLite), kept separate from the Mem0 memory store.

The memory store holds text that a model may read. The register holds what the
application *knows* about where that text came from: authenticated source identity,
the role the harness assigned, the role's permissions at the time, the original message,
registered approvals, and lineage links from messages to memory record IDs.

Rules enforced here:
* Roles come from the envelope supplied by the harness, never from message text.
* A message that *claims* an approval never creates an approval record; approvals are
  registered separately and explicitly (``register_approval``).
* Lineage is recorded from observed store changes. When a record may derive from more
  than one source, its status is ``ambiguous`` and its effective authority is the
  *least* privileged candidate. Records with no recorded lineage are ``unknown`` and have
  no authority. Nothing is ever promoted to a more privileged role.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas.scenario import Approval, RoleCatalog

PROVENANCE_STATUSES = ("direct", "attributed", "ambiguous", "unknown")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS envelopes (
  message_id TEXT NOT NULL, namespace TEXT NOT NULL, source_id TEXT NOT NULL,
  display_name TEXT NOT NULL, role TEXT NOT NULL, permissions TEXT NOT NULL,
  max_exception_amount_pkr INTEGER, original_message TEXT NOT NULL, received_at TEXT NOT NULL,
  policy_scope TEXT NOT NULL, claimed_authorization TEXT, seq INTEGER NOT NULL,
  PRIMARY KEY (message_id, namespace)
);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT NOT NULL, namespace TEXT NOT NULL, body TEXT NOT NULL,
  PRIMARY KEY (approval_id, namespace)
);
CREATE TABLE IF NOT EXISTS lineage (
  record_id TEXT NOT NULL, namespace TEXT NOT NULL, primary_message_id TEXT,
  candidate_message_ids TEXT NOT NULL, status TEXT NOT NULL, text_at_link TEXT NOT NULL,
  deleted INTEGER NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (record_id, namespace)
);
CREATE TABLE IF NOT EXISTS record_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL, record_id TEXT NOT NULL,
  event TEXT NOT NULL, message_id TEXT, old_text TEXT, new_text TEXT, at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class EffectiveSource:
    status: str
    primary_message_id: str | None
    primary_source_id: str | None
    primary_display_name: str | None
    primary_role: str | None
    candidate_roles: tuple[str, ...]
    effective_role: str            # least-privileged candidate role, or "unknown"
    effective_permissions: tuple[str, ...]
    effective_max_amount_pkr: int | None


class SourceRegister:
    def __init__(self, path: Path, roles: RoleCatalog):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.roles = roles
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ---- envelopes & approvals --------------------------------------------------------
    def register_envelope(self, *, namespace: str, message_id: str, source_id: str, display_name: str,
                          role: str, original_message: str, policy_scope: str,
                          claimed_authorization: dict | None = None, received_at: str | None = None) -> None:
        r = self.roles.get(role)  # KeyError for unknown roles: roles must be defined in the fixture
        seq = self.db.execute("SELECT COUNT(*) FROM envelopes WHERE namespace=?", (namespace,)).fetchone()[0]
        self.db.execute(
            "INSERT INTO envelopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (message_id, namespace, source_id, display_name, role, json.dumps(list(r.permissions)),
             r.max_exception_amount_pkr, original_message,
             received_at or dt.datetime.now(dt.timezone.utc).isoformat(), policy_scope,
             json.dumps(claimed_authorization, default=str) if claimed_authorization else None, seq))
        self.db.commit()

    def register_approval(self, namespace: str, approval: Approval) -> None:
        self.db.execute("INSERT INTO approvals VALUES (?,?,?)",
                        (approval.approval_id, namespace, approval.model_dump_json()))
        self.db.commit()

    def approvals(self, namespace: str) -> tuple[Approval, ...]:
        rows = self.db.execute("SELECT body FROM approvals WHERE namespace=? ORDER BY approval_id", (namespace,))
        return tuple(Approval.model_validate_json(r["body"]) for r in rows)

    def envelope(self, namespace: str, message_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM envelopes WHERE namespace=? AND message_id=?",
                              (namespace, message_id)).fetchone()
        return dict(row) if row else None

    def envelopes(self, namespace: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM envelopes WHERE namespace=? ORDER BY seq", (namespace,))
        return [dict(r) for r in rows]

    # ---- lineage ----------------------------------------------------------------------
    def _event(self, namespace, record_id, event, message_id, old, new) -> None:
        self.db.execute("INSERT INTO record_events (namespace, record_id, event, message_id, old_text, new_text, at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (namespace, record_id, event, message_id, old, new, dt.datetime.now(dt.timezone.utc).isoformat()))

    def link_created(self, namespace: str, record_id: str, text: str, primary_message_id: str,
                     candidate_message_ids: list[str], status: str, note: str = "") -> None:
        assert status in PROVENANCE_STATUSES
        cands = sorted(set(candidate_message_ids) | {primary_message_id})
        self.db.execute("INSERT OR REPLACE INTO lineage VALUES (?,?,?,?,?,?,0,?)",
                        (record_id, namespace, primary_message_id, json.dumps(cands), status, text, note))
        self._event(namespace, record_id, "CREATE", primary_message_id, None, text)
        self.db.commit()

    def link_updated(self, namespace: str, record_id: str, old: str, new: str, message_id: str | None) -> None:
        """An existing record changed. Its content may now mix sources -> ambiguous (or unknown)."""
        row = self.db.execute("SELECT * FROM lineage WHERE record_id=? AND namespace=?",
                              (record_id, namespace)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO lineage VALUES (?,?,?,?,?,?,0,?)",
                            (record_id, namespace, None, json.dumps([message_id] if message_id else []),
                             "unknown", new, "updated record with no prior lineage"))
        else:
            cands = set(json.loads(row["candidate_message_ids"]))
            if message_id:
                cands.add(message_id)
            same_single_source = (
                message_id is not None and row["status"] in ("direct", "attributed")
                and len({self.envelope(namespace, m)["source_id"] for m in cands if self.envelope(namespace, m)}) == 1)
            status = row["status"] if same_single_source else "ambiguous"
            self.db.execute("UPDATE lineage SET candidate_message_ids=?, status=?, text_at_link=?, note=? "
                            "WHERE record_id=? AND namespace=?",
                            (json.dumps(sorted(cands)), status, new, "content changed after creation",
                             record_id, namespace))
        self._event(namespace, record_id, "UPDATE", message_id, old, new)
        self.db.commit()

    def link_deleted(self, namespace: str, record_id: str, text: str, message_id: str | None) -> None:
        self.db.execute("UPDATE lineage SET deleted=1 WHERE record_id=? AND namespace=?", (record_id, namespace))
        self._event(namespace, record_id, "DELETE", message_id, text, None)
        self.db.commit()

    def apply_changeset(self, namespace: str, message_id: str, changes: dict, context_message_ids: list[str],
                        inferred: bool) -> None:
        """Record lineage for every change observed while processing ``message_id``.

        ``context_message_ids``: earlier messages in the same memory scope that the memory
        system could have used as context. If any came from a different source, records
        created now are ``ambiguous``.
        """
        own = self.envelope(namespace, message_id)
        ctx_sources = {self.envelope(namespace, m)["source_id"] for m in context_message_ids if self.envelope(namespace, m)}
        mixed = inferred and bool(ctx_sources - {own["source_id"]})
        for c in changes["created"]:
            if not inferred:
                self.link_created(namespace, c["id"], c["text"], message_id, [], "direct")
            elif mixed:
                self.link_created(namespace, c["id"], c["text"], message_id, context_message_ids, "ambiguous",
                                  "LLM extraction saw context from other sources")
            else:
                self.link_created(namespace, c["id"], c["text"], message_id, context_message_ids, "attributed")
        for u in changes["updated"]:
            self.link_updated(namespace, u["id"], u["old"], u["new"], message_id)
        for d in changes["deleted"]:
            self.link_deleted(namespace, d["id"], d["text"], message_id)

    def lineage(self, namespace: str, record_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM lineage WHERE record_id=? AND namespace=?",
                              (record_id, namespace)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["candidate_message_ids"] = json.loads(d["candidate_message_ids"])
        return d

    def records_from_message(self, namespace: str, message_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM lineage WHERE namespace=? AND primary_message_id=?",
                               (namespace, message_id))
        return [dict(r) for r in rows]

    def events(self, namespace: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM record_events WHERE namespace=? ORDER BY id", (namespace,))
        return [dict(r) for r in rows]

    # ---- authority --------------------------------------------------------------------
    def _least_privileged(self, roles: list[str]) -> str:
        def rank(role: str) -> tuple[int, int]:
            r = self.roles.get(role)
            return (len(r.permissions), r.max_exception_amount_pkr or 0)
        return min(roles, key=rank)

    def effective_source(self, namespace: str, record_id: str) -> EffectiveSource:
        lin = self.lineage(namespace, record_id)
        if lin is None or lin["status"] == "unknown" or not lin["primary_message_id"]:
            return EffectiveSource("unknown", None, None, None, None, (), "unknown", (), None)
        prim = self.envelope(namespace, lin["primary_message_id"])
        cand_roles = [self.envelope(namespace, m)["role"] for m in lin["candidate_message_ids"] if self.envelope(namespace, m)]
        if lin["status"] in ("direct", "attributed"):
            eff_role = prim["role"]
        else:  # ambiguous: never promote; use the least privileged possible contributor
            eff_role = self._least_privileged(cand_roles or [prim["role"]])
        r = self.roles.get(eff_role)
        return EffectiveSource(lin["status"], prim["message_id"], prim["source_id"], prim["display_name"],
                               prim["role"], tuple(sorted(set(cand_roles))), eff_role, r.permissions,
                               r.max_exception_amount_pkr)

    def export(self) -> dict[str, Any]:
        out = {}
        for table in ("envelopes", "approvals", "lineage", "record_events"):
            out[table] = [dict(r) for r in self.db.execute(f"SELECT * FROM {table}")]
        return out
