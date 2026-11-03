"""Per-run results store: SQLite (queryable) + one JSON trace file per attempt (portable).

Every attempt is kept. Nothing is overwritten: a retry is a new row with attempt = n + 1.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..schemas.records import TrialResult

FINISHED_STATUSES = ("completed", "invalid_output", "step_limit", "model_refusal")  # model gave an answer
FAILED_STATUSES = ("api_error",)
INCOMPLETE_STATUSES = ("budget_stop",)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
  trial_id TEXT NOT NULL, attempt INTEGER NOT NULL, run_id TEXT NOT NULL, mode TEXT NOT NULL,
  experiment_kind TEXT NOT NULL, scenario_id TEXT NOT NULL, family TEXT NOT NULL, condition TEXT NOT NULL,
  config TEXT NOT NULL, repetition INTEGER NOT NULL, status TEXT NOT NULL, terminal_action TEXT,
  success INTEGER, prohibited_attempt INTEGER, unnecessary_approval INTEGER, refusal INTEGER,
  valid_output INTEGER, policy_visible INTEGER, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, duration_s REAL, labels TEXT, started_at TEXT, finished_at TEXT, trace_file TEXT,
  PRIMARY KEY (trial_id, attempt)
);
"""


class ResultStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "traces").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / "results.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    def next_attempt(self, trial_id: str) -> int:
        row = self.db.execute("SELECT MAX(attempt) FROM attempts WHERE trial_id=?", (trial_id,)).fetchone()
        return (row[0] or 0) + 1

    def latest(self, trial_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM attempts WHERE trial_id=? ORDER BY attempt DESC LIMIT 1",
                              (trial_id,)).fetchone()
        return dict(row) if row else None

    def save(self, r: TrialResult) -> None:
        trace = f"traces/{r.trial_id}__a{r.attempt}.json"
        tmp = self.run_dir / (trace + ".tmp")
        tmp.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.run_dir / trace)
        s = r.score
        self.db.execute(
            "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.trial_id, r.attempt, r.run_id, r.mode, r.experiment_kind, r.scenario_id, r.family, r.condition,
             r.config, r.repetition, s.status, s.terminal_action, int(s.success), int(s.prohibited_attempt),
             int(s.unnecessary_approval), int(s.refusal), int(s.valid_output), int(r.policy_visible),
             r.input_tokens, r.output_tokens, r.cost_usd, r.duration_s, json.dumps(r.labels), r.started_at,
             r.finished_at, trace))
        self.db.commit()

    def all_attempts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM attempts ORDER BY trial_id, attempt")]

    def latest_attempts(self) -> list[dict[str, Any]]:
        q = ("SELECT a.* FROM attempts a JOIN (SELECT trial_id, MAX(attempt) m FROM attempts GROUP BY trial_id) b "
             "ON a.trial_id=b.trial_id AND a.attempt=b.m ORDER BY a.trial_id")
        return [dict(r) for r in self.db.execute(q)]

    def load_trace(self, trace_file: str) -> dict[str, Any]:
        return json.loads((self.run_dir / trace_file).read_text(encoding="utf-8"))
