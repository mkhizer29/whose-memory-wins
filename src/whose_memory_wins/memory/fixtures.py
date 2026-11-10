"""Track 1: controlled memory fixtures and frozen retrieval snapshots.

For each scenario x condition, a separate Mem0 namespace (``user_id``) is filled with known
records through Mem0's documented ``add(..., infer=False)`` path and verified by readback.
The application source register records the envelope (source, role, permissions) of each
input message and the lineage message -> memory record.

Retrieval is then performed ONCE per scenario x condition with the task's fixed query and
saved as an immutable JSON snapshot. Every trial for that scenario x condition (A, B, C and
all repetitions) reads the same snapshot, so the base retrieval is identical across
configurations by construction.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
from pathlib import Path
from typing import Any

from ..dataset import load_roles, load_split, split_hash, stable_hash
from ..policies.engine import evaluate_policies
from ..provenance.register import SourceRegister
from ..schemas.scenario import Condition, RoleCatalog, Scenario
from ..snapshots import finalize_snapshot, write_snapshot
from .mem0_store import Mem0Store


def namespace(scenario_id: str, condition: str) -> str:
    return f"{scenario_id}__{condition}"


def applicable_policy_ids(scenario: Scenario) -> list[str]:
    """Policies that bind the task; for permitted tasks, the policies of the scenario's family."""
    t = scenario.task
    d = evaluate_policies(scenario, t.supplier_id, t.total_pkr)
    ids = [r.policy.policy_id for r in d.approval_needed + d.prohibited]
    return ids or [p.policy_id for p in scenario.policies if p.family == scenario.family]


FIXTURE_LAYOUT_VERSION = 2  # v2: one Mem0 store per namespace (see namespace_store_dir)


def fixture_id(split: str, retrieval: dict) -> str:
    return f"{split}-{split_hash(split)[:10]}-{stable_hash([retrieval, FIXTURE_LAYOUT_VERSION])[:6]}"


def fixture_dir(outputs: Path, fid: str) -> Path:
    return outputs / "fixtures" / "controlled" / fid


def namespace_store_dir(fdir: Path, ns: str) -> Path:
    """One physically separate Mem0 store per scenario x condition.

    Mem0's Qdrant keyword channel (BM25 with IDF) uses collection-wide statistics, so sharing one
    collection across namespaces would make a namespace's ranking depend on other namespaces' records.
    """
    return fdir / "mem0" / ns


def populate_namespace(store: Mem0Store, register: SourceRegister, scenario: Scenario, condition: Condition,
                       ns: str) -> list[dict[str, Any]]:
    """Insert facts + condition messages verbatim; register envelopes, approvals and lineage."""
    inserted = []
    base_time = dt.datetime.combine(scenario.company.current_date, dt.time(9, 0), tzinfo=dt.timezone.utc)
    items = [(f.record_key, f.source_id, f.message_text, f.memory_text, f.kind, f.policy_id, None)
             for f in scenario.facts]
    spec = scenario.conditions[condition]
    for m in spec.messages:
        items.append((m.message_id, m.source_id, m.message_text, m.memory_text, "claim", None,
                      m.claimed_authorization.model_dump(mode="json") if m.claimed_authorization else None))
    for i, (key, source_id, message_text, memory_text, kind, policy_id, claimed) in enumerate(items):
        src = scenario.source(source_id)
        message_id = key if key.startswith("MSG-") else f"MSG-{key}"
        register.register_envelope(
            namespace=ns, message_id=message_id, source_id=src.source_id, display_name=src.display_name,
            role=src.role, original_message=message_text, policy_scope=f"company:{scenario.company.name}",
            claimed_authorization=claimed, received_at=(base_time + dt.timedelta(minutes=i)).isoformat())
        meta = {"wmw_record_key": key, "wmw_message_id": message_id, "wmw_kind": kind}
        if policy_id:
            meta["wmw_policy_id"] = policy_id
        rid, cs = store.insert_verbatim(ns, memory_text, meta)
        register.apply_changeset(ns, message_id, cs.to_dict(), context_message_ids=[], inferred=False)
        inserted.append({"record_id": rid, "record_key": key, "message_id": message_id, "kind": kind,
                         "policy_id": policy_id, "text": memory_text})
    for a in spec.approvals:
        register.register_approval(ns, a)
    return inserted


def snapshot_retrieval(store: Mem0Store, register: SourceRegister, scenario: Scenario, condition: str, ns: str,
                       retrieval: dict, extra: dict | None = None) -> dict[str, Any]:
    hits = store.search(ns, scenario.task.retrieval_query, retrieval["top_k"], retrieval["threshold"])
    approvals = register.approvals(ns)
    all_recs = store.records(ns)
    claim_ids = []
    for rid, rec in all_recs.items():
        lin = register.lineage(ns, rid)
        if lin and lin["primary_message_id"]:
            env = register.envelope(ns, lin["primary_message_id"])
            if env and env["claimed_authorization"]:
                claim_ids.append(rid)
    records = []
    for h in hits:
        eff = register.effective_source(ns, h["id"])
        lin = register.lineage(ns, h["id"]) or {}
        linked = [a.model_dump(mode="json") for a in approvals
                  if a.source_message_id in (lin.get("candidate_message_ids") or [])
                  and eff.primary_message_id == a.source_message_id]
        records.append({
            **h,
            "provenance_status": eff.status,
            "source": {
                "primary_message_id": eff.primary_message_id, "source_id": eff.primary_source_id,
                "display_name": eff.primary_display_name, "assigned_role": eff.primary_role,
                "candidate_roles": list(eff.candidate_roles), "effective_role": eff.effective_role,
                "effective_permissions": list(eff.effective_permissions),
                "effective_max_amount_pkr": eff.effective_max_amount_pkr,
            },
            "linked_approvals": linked,
        })
    retrieved_ids = {r["id"] for r in records}
    app_ids = applicable_policy_ids(scenario)
    snap = {
        "scenario_id": scenario.scenario_id, "condition": condition, "namespace": ns,
        "query": scenario.task.retrieval_query, "top_k": retrieval["top_k"], "threshold": retrieval["threshold"],
        "records": records,
        "namespace_record_count": len(all_recs),
        "namespace_digest": store.namespace_digest(ns),
        "claim_record_ids": sorted(claim_ids),
        "claim_present_in_memory": bool(claim_ids),
        "claim_retrieved": bool(retrieved_ids & set(claim_ids)),
        "claim_records": [{"id": r["id"], "provenance_status": r["provenance_status"]}
                          for r in records if r["id"] in claim_ids],
        "applicable_policy_ids": app_ids,
        "policy_retrieved": any(r["metadata"].get("wmw_policy_id") in app_ids for r in records),
        "registered_approval_ids": [a.approval_id for a in approvals],
    }
    if extra:
        snap.update(extra)
    return finalize_snapshot(snap)  # canonical content hash; must stay the last change to `snap`


def build_controlled_fixture(split: str, outputs: Path, retrieval: dict, rebuild: bool = False,
                             roles: RoleCatalog | None = None, scenario_ids: list[str] | None = None,
                             log=print) -> Path:
    roles = roles or load_roles()
    fid = fixture_id(split, retrieval)
    fdir = fixture_dir(outputs, fid)
    manifest_path = fdir / "fixture_manifest.json"
    if manifest_path.exists() and not rebuild:
        log(f"Fixture {fid} already built (immutable); reusing {fdir}")
        return fdir
    if fdir.exists():
        import shutil
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    scenarios, _ = load_split(split)
    if scenario_ids:
        scenarios = [s for s in scenarios if s.scenario_id in scenario_ids]
    register = SourceRegister(fdir / "source_register.sqlite", roles)
    (fdir / "retrieval").mkdir()
    inserted_all = {}
    for s in scenarios:
        for cond in s.conditions:
            ns = namespace(s.scenario_id, cond.value)
            store = Mem0Store(namespace_store_dir(fdir, ns), retrieval)
            try:
                inserted_all[ns] = populate_namespace(store, register, s, cond, ns)
                snap = snapshot_retrieval(store, register, s, cond.value, ns, retrieval)
            finally:
                store.close()
            write_snapshot(fdir / "retrieval" / f"{ns}.json", snap)
        log(f"  fixture: {s.scenario_id} ({len(s.conditions)} conditions)")
    (fdir / "inserted_records.json").write_text(json.dumps(inserted_all, indent=2), encoding="utf-8")
    (fdir / "source_register_export.json").write_text(json.dumps(register.export(), indent=2, default=str),
                                                      encoding="utf-8")
    register.close()
    manifest = {
        "fixture_id": fid, "track": "controlled_insertion", "split": split,
        "dataset_hash": split_hash(split), "retrieval": retrieval,
        "insertion_mechanism": "mem0.Memory.add(messages, user_id=..., metadata=..., infer=False) "
                               "+ readback via Memory.get()",
        "bypassed_components": ["Mem0 LLM fact extraction (infer=True path) is not used in Track 1"],
        "mem0_version": md.version("mem0ai"), "fastembed_version": md.version("fastembed"),
        "qdrant_client_version": md.version("qdrant-client"),
        "spacy_installed": _has("spacy"),
        "namespaces": sorted(inserted_all),
        "store_layout": "one Mem0/Qdrant store per namespace (mem0/<namespace>/) for retrieval isolation",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "immutable": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Built fixture {fid} at {fdir}")
    return fdir


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def load_snapshot(fdir: Path, scenario_id: str, condition: str) -> dict[str, Any]:
    return json.loads((fdir / "retrieval" / f"{namespace(scenario_id, condition)}.json").read_text(encoding="utf-8"))
