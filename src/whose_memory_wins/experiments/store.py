"""Per-run results store: SQLite (queryable) + one JSON trace file per attempt (portable).

Every attempt is kept. Nothing is overwritten: a retry is a new row with attempt = n + 1.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..schemas.records import TrialResult

# Statuses where the model gave an answer (a decision, a non-decision, or a bounded failure).
FINISHED_STATUSES = ("completed", "no_decision", "invalid_output", "step_limit", "model_refusal")
FAILED_STATUSES = ("api_error",)
INCOMPLETE_STATUSES = ("budget_stop",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
  trial_id TEXT NOT NULL, attempt INTEGER NOT NULL, run_id TEXT NOT NULL, mode TEXT NOT NULL,
  experiment_kind TEXT NOT NULL, scenario_id TEXT NOT NULL, family TEXT NOT NULL, condition TEXT NOT NULL,
  config TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL, terminal_action TEXT,
  success INTEGER, prohibited_attempt INTEGER, unnecessary_approval INTEGER, refusal INTEGER,
  valid_decision INTEGER, policy_visible INTEGER, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, duration_s REAL, labels TEXT, started_at TEXT, finished_at TEXT, trace_file TEXT,
  scoring_version TEXT, claim_status TEXT, cited_claim_reference INTEGER,
  PRIMARY KEY (trial_id, attempt)
);
"""
# Columns added in scoring version 2. Databases written by version 1 lack them (and have
# ``valid_output`` instead, which meant only "status == completed"). They are read, never altered.
V2_COLUMNS = ("valid_decision", "scoring_version", "claim_status", "cited_claim_reference")


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("scoring_version") is None:
        row["scoring_version"] = "1"          # written before scoring versions existed
    for col in V2_COLUMNS:
        row.setdefault(col, None)
    return row


class ResultStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "traces").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / "results.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.columns = {r[1] for r in self.db.execute("PRAGMA table_info(attempts)")}
        self.legacy = not set(V2_COLUMNS) <= self.columns

    def close(self) -> None:
        self.db.close()

    def next_attempt(self, trial_id: str) -> int:
        row = self.db.execute("SELECT MAX(attempt) FROM attempts WHERE trial_id=?", (trial_id,)).fetchone()
        return (row[0] or 0) + 1

    def latest(self, trial_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM attempts WHERE trial_id=? ORDER BY attempt DESC LIMIT 1",
                              (trial_id,)).fetchone()
        return _normalize(dict(row)) if row else None

    def save(self, r: TrialResult) -> None:
        if self.legacy:
            raise RuntimeError(f"{self.run_dir} holds results scored with older rules (scoring version 1); "
                               "new results are never mixed into it. Create a new run.")
        trace = f"traces/{r.trial_id}__a{r.attempt}.json"
        tmp = self.run_dir / (trace + ".tmp")
        tmp.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.run_dir / trace)
        s = r.score
        row = {"trial_id": r.trial_id, "attempt": r.attempt, "run_id": r.run_id, "mode": r.mode,
               "experiment_kind": r.experiment_kind, "scenario_id": r.scenario_id, "family": r.family,
               "condition": r.condition, "config": r.config, "repetition": r.repetition, "status": s.status,
               "terminal_action": s.terminal_action, "success": int(s.success),
               "prohibited_attempt": int(s.prohibited_attempt), "unnecessary_approval": int(s.unnecessary_approval),
               "refusal": int(s.refusal), "valid_decision": int(s.valid_decision),
               "policy_visible": int(r.policy_visible), "input_tokens": r.input_tokens,
               "output_tokens": r.output_tokens, "cost_usd": r.cost_usd, "duration_s": r.duration_s,
               "labels": json.dumps(r.labels), "started_at": r.started_at, "finished_at": r.finished_at,
               "trace_file": trace, "scoring_version": s.scoring_version,
               "claim_status": r.evidence.get("claim_status"),
               "cited_claim_reference": int(bool(r.evidence.get("cited_claim_reference")))}
        cols = ",".join(row)
        self.db.execute(f"INSERT INTO attempts ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        self.db.commit()

    def all_attempts(self) -> list[dict[str, Any]]:
        return [_normalize(dict(r)) for r in self.db.execute("SELECT * FROM attempts ORDER BY trial_id, attempt")]

    def latest_attempts(self) -> list[dict[str, Any]]:
        q = ("SELECT a.* FROM attempts a JOIN (SELECT trial_id, MAX(attempt) m FROM attempts GROUP BY trial_id) b "
             "ON a.trial_id=b.trial_id AND a.attempt=b.m ORDER BY a.trial_id")
        return [_normalize(dict(r)) for r in self.db.execute(q)]

    def load_trace(self, trace_file: str) -> dict[str, Any]:
        return json.loads((self.run_dir / trace_file).read_text(encoding="utf-8"))
