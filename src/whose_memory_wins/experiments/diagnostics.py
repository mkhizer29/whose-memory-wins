"""Optional memory-removal reruns for selected cases.

A diagnostic run copies the frozen retrieval snapshot of one scenario x condition, removes
chosen records (by memory rank label M<n> or by kind, e.g. ``claim``), and creates a SEPARATE
run of kind ``diagnostic_removal``. Original results are never replaced. Removed records
are not back-filled with the next search hit (the retrieval is not re-run).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..config import outputs_dir
from ..snapshots import verify_snapshot_file, write_snapshot
from .runner import create_run, load_manifest


def create_removal_run(source_run: Path, scenario_id: str, condition: str, remove: list[str],
                       configs: list[str] | None = None, reps: int = 1, seed: int = 7) -> Path:
    """Create a memory-removal diagnostic run from an existing run's snapshot.

    The source snapshot is verified against the source manifest first. The chosen
    records (claim and/or policy) are removed from a COPY, which is re-hashed and
    used for a new, separate run. The original snapshot is never modified.
    """
    m = load_manifest(source_run)
    cfg = m["config"]
    snap_file = Path(m["snapshot_dir"]) / f"{scenario_id}__{condition}.json"
    expected = next((t["snapshot_hash"] for t in m["trials"]
                     if t["scenario_id"] == scenario_id and t["condition"] == condition), None)
    snap = verify_snapshot_file(snap_file, expected)  # never derive a diagnostic from a tampered snapshot
    keep, removed = [], []
    for i, r in enumerate(snap["records"]):
        label = f"M{i + 1}"
        kind = r["metadata"].get("wmw_kind")
        (removed if (label in remove or kind in remove) else keep).append(r)
    if not removed:
        raise ValueError(f"nothing matched {remove}; use labels like M3 or kinds like claim/policy")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ddir = outputs_dir(cfg) / "diagnostics" / f"{m['mode']}-removal-{scenario_id}-{condition}-{stamp}"
    (ddir / "retrieval").mkdir(parents=True)
    new = dict(snap)
    new["records"] = keep
    claim_ids = set(snap["claim_record_ids"])
    new["claim_retrieved"] = any(r["id"] in claim_ids for r in keep)
    new["claim_records"] = [c for c in snap["claim_records"] if c["id"] in {r["id"] for r in keep}]
    new["policy_retrieved"] = any(r["metadata"].get("wmw_policy_id") in snap["applicable_policy_ids"] for r in keep)
    new["diagnostic_removed"] = [{"id": r["id"], "text": r["text"]} for r in removed]
    new["derived_from_snapshot"] = snap["snapshot_hash"]
    write_snapshot(ddir / "retrieval" / snap_file.name, new)  # canonical hash of the final content
    return create_run(cfg, name=f"removal-{scenario_id}", split=m["split"], experiment_kind="diagnostic_removal",
                      snapshot_dir=ddir / "retrieval", register_path=Path(m["register_path"]),
                      scenario_ids=[scenario_id], conditions=[condition], configs=configs or m["configs"],
                      reps=reps, seed=seed, run_root=ddir,
                      extra={"diagnostic_of_run": m["run_id"], "removed_records": new["diagnostic_removed"]})
