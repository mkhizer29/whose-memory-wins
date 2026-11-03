"""Loading, hashing and validating the scenario dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .env import PROJECT_ROOT
from .policies.engine import evaluate_policies, oracle_terminal, to_paisa
from .schemas.ground_truth import ScenarioTruth
from .schemas.scenario import (
    CORE_CONDITIONS,
    Condition,
    Family,
    RoleCatalog,
    Scenario,
    SpendingRule,
)

DATASETS = PROJECT_ROOT / "datasets"
SPLITS = ("development", "evaluation")


def roles_path() -> Path:
    return DATASETS / "roles.json"


def load_roles(path: Path | None = None) -> RoleCatalog:
    return RoleCatalog.model_validate_json((path or roles_path()).read_text(encoding="utf-8"))


def load_split(split: str, root: Path | None = None) -> tuple[list[Scenario], dict[str, ScenarioTruth]]:
    base = (root or DATASETS) / split
    scenarios = [Scenario.model_validate_json(p.read_text(encoding="utf-8"))
                 for p in sorted((base / "scenarios").glob("*.json"))]
    truths = {}
    for p in sorted((base / "ground_truth").glob("*.json")):
        t = ScenarioTruth.model_validate_json(p.read_text(encoding="utf-8"))
        truths[t.scenario_id] = t
    return scenarios, truths


def load_scenario(scenario_id: str) -> tuple[Scenario, ScenarioTruth]:
    for split in SPLITS:
        p = DATASETS / split / "scenarios" / f"{scenario_id}.json"
        if p.exists():
            s = Scenario.model_validate_json(p.read_text(encoding="utf-8"))
            t = ScenarioTruth.model_validate_json(
                (DATASETS / split / "ground_truth" / f"{scenario_id}.json").read_text(encoding="utf-8"))
            return s, t
    raise KeyError(scenario_id)


def split_hash(split: str, root: Path | None = None) -> str:
    """SHA-256 over every file of the split plus the role catalog (order-stable)."""
    base = (root or DATASETS)
    h = hashlib.sha256()
    files = sorted((base / split).rglob("*.json")) + [base / "roles.json"]
    for p in files:
        h.update(p.relative_to(base).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


GROUND_TRUTH_MARKERS = ("acceptable_terminal_actions", "prohibited_kinds", "expected", "ground truth", "conflict_relevant")


def validate_split(split: str, scenarios: list[Scenario], truths: dict[str, ScenarioTruth],
                   roles: RoleCatalog) -> list[str]:
    errs: list[str] = []
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        errs.append("duplicate scenario IDs")
    if set(ids) != set(truths):
        errs.append(f"scenario/ground-truth ID mismatch: {sorted(set(ids) ^ set(truths))}")
    all_msg_ids: list[str] = []

    if split == "evaluation":
        if len(scenarios) != 24:
            errs.append(f"evaluation split must have 24 scenarios, found {len(scenarios)}")
        for fam in Family:
            n = sum(s.family == fam for s in scenarios)
            if n != 12:
                errs.append(f"evaluation split must have 12 {fam.value} scenarios, found {n}")
    if split == "development" and len(scenarios) != 6:
        errs.append(f"development split must have 6 scenarios, found {len(scenarios)}")

    clean_kinds = set()
    boundary_exact = boundary_above = False
    for s in scenarios:
        p = f"{s.scenario_id}:"
        if s.split != split:
            errs.append(f"{p} split field {s.split!r} != {split!r}")
        raw = s.model_dump_json().lower()
        for marker in GROUND_TRUTH_MARKERS:
            if marker in raw:
                errs.append(f"{p} model-side scenario file contains evaluator marker {marker!r}")
        t = s.task
        item = s.item(t.supplier_id, t.item_id)
        if item is None:
            errs.append(f"{p} task item {t.item_id} not sold by {t.supplier_id}")
        else:
            if item.unit_price_pkr != t.unit_price_pkr:
                errs.append(f"{p} task unit price differs from catalogue")
            if item.stock < t.quantity:
                errs.append(f"{p} insufficient stock makes the task unsolvable")
        if to_paisa(t.unit_price_pkr) * t.quantity != to_paisa(t.total_pkr):
            errs.append(f"{p} total_pkr != quantity * unit_price")
        for tok in (t.supplier_id, t.item_id, f"{t.total_pkr:,}"):
            if tok not in t.text:
                errs.append(f"{p} task text must state {tok}")
        if not t.retrieval_query.strip():
            errs.append(f"{p} empty retrieval query")
        src_ids = {x.source_id for x in s.sources}
        for src in s.sources:
            try:
                roles.get(src.role)
            except KeyError:
                errs.append(f"{p} source {src.source_id} has undefined role {src.role}")
        keys = [f.record_key for f in s.facts]
        if len(keys) != len(set(keys)):
            errs.append(f"{p} duplicate fact record keys")
        for f in s.facts:
            if f.source_id not in src_ids:
                errs.append(f"{p} fact {f.record_key} has unknown source")
        for pol in s.policies:
            if pol.rule.type == "prohibited_suppliers" and pol.overridable:
                errs.append(f"{p} policy {pol.policy_id}: prohibited-supplier rules must be non-overridable")
        if not any(f.kind == "policy" and f.policy_id == pol.policy_id for pol in s.policies for f in s.facts):
            errs.append(f"{p} policies must also exist as memory facts (eligible for ordinary retrieval)")
        if not any(pol.family == s.family for pol in s.policies):
            errs.append(f"{p} no policy of the scenario family")

        truth = truths.get(s.scenario_id)
        if truth is None:
            continue
        if split == "evaluation" and set(s.conditions) != set(CORE_CONDITIONS):
            errs.append(f"{p} evaluation scenarios need exactly clean/untrusted/authorized conditions")
        if set(truth.conditions) != set(s.conditions):
            errs.append(f"{p} ground truth conditions differ from scenario conditions")

        decision = evaluate_policies(s, t.supplier_id, t.total_pkr)
        for spending in (pol.rule for pol in s.policies if isinstance(pol.rule, SpendingRule)):
            if t.total_pkr == spending.threshold_pkr:
                boundary_exact = True
            if 0 < t.total_pkr - spending.threshold_pkr <= spending.threshold_pkr // 100:
                boundary_above = True

        for cond, spec in s.conditions.items():
            cp = f"{p}{cond.value}:"
            ct = truth.conditions.get(cond)
            for m in spec.messages:
                all_msg_ids.append(m.message_id)
                if m.source_id not in src_ids:
                    errs.append(f"{cp} message from unknown source {m.source_id}")
            for a in spec.approvals:
                if a.source_message_id not in {m.message_id for m in spec.messages}:
                    errs.append(f"{cp} approval {a.approval_id} not linked to a condition message")
                if a.granted_by not in src_ids:
                    errs.append(f"{cp} approval grantor unknown")
            if ct is None:
                continue
            if ct.requires_approval_absent_exception != decision.requires_approval:
                errs.append(f"{cp} requires_approval_absent_exception disagrees with the policy engine")
            oracle, ref = oracle_terminal(s, spec.approvals, roles)
            if ct.valid_exception_present != (ref is not None):
                errs.append(f"{cp} valid_exception_present disagrees with the policy engine")
            match = [a for a in ct.acceptable_terminal_actions if a.tool == oracle]
            if not match:
                errs.append(f"{cp} policy-engine answer {oracle} is not among the acceptable actions")
            for a in ct.acceptable_terminal_actions:
                if a.tool in ("place_order", "request_approval") and (
                        (a.supplier_id, a.item_id, a.quantity, a.amount_pkr)
                        != (t.supplier_id, t.item_id, t.quantity, t.total_pkr)):
                    errs.append(f"{cp} acceptable action arguments do not match the task")
                if a.tool == "place_order" and decision.requires_approval and not (
                        a.reference_rule == "must_equal" and a.approval_reference == ref):
                    errs.append(f"{cp} acceptable place_order must require the valid approval reference")
                if a.tool == "request_approval" and not decision.requires_approval and not decision.prohibited:
                    errs.append(f"{cp} request_approval marked acceptable for a permitted purchase")
            if cond == Condition.CLEAN:
                if spec.messages or spec.approvals:
                    errs.append(f"{cp} clean condition must have no claim messages or approvals")
                clean_kinds.add("approval" if decision.requires_approval else
                                ("prohibited" if decision.prohibited else "permitted"))
            if cond in (Condition.UNTRUSTED, Condition.IMPERSONATION):
                if spec.approvals:
                    errs.append(f"{cp} untrusted claims must not create approval records")
                for m in spec.messages:
                    if roles.get(s.source(m.source_id).role).permissions:
                        errs.append(f"{cp} untrusted claim sender must have no authorization permissions")
            if cond == Condition.AUTHORIZED:
                if any(not r.policy.overridable for r in decision.approval_needed) or decision.prohibited:
                    errs.append(f"{cp} authorized exception targets a non-overridable rule")
                if truth.conflict_relevant and ref is None:
                    errs.append(f"{cp} authorized exception is not valid (unsolvable condition)")
                unt = s.conditions.get(Condition.UNTRUSTED)
                if unt is not None:
                    if [m.memory_text for m in unt.messages] != [m.memory_text for m in spec.messages]:
                        errs.append(f"{cp} claim memory text must match the untrusted condition")
                    if [m.message_text for m in unt.messages] != [m.message_text for m in spec.messages]:
                        errs.append(f"{cp} claim message text must match the untrusted condition")
                    if {m.source_id for m in unt.messages} & {m.source_id for m in spec.messages}:
                        errs.append(f"{cp} authorized and untrusted claims must come from different sources")
        if truth.conflict_relevant != decision.requires_approval:
            errs.append(f"{p} conflict_relevant must equal 'task requires approval absent an exception'")

    if len(all_msg_ids) != len(set(all_msg_ids)):
        errs.append("duplicate message IDs across the split")
    if split == "evaluation":
        if not {"approval", "permitted"} <= clean_kinds:
            errs.append("clean conditions must include both permitted purchases and approval-required purchases")
        if not (boundary_exact and boundary_above):
            errs.append("evaluation split needs an exactly-at-threshold and a just-above-threshold case")
    return errs


def validate_all() -> dict[str, list[str]]:
    roles = load_roles()
    out = {}
    for split in SPLITS:
        scenarios, truths = load_split(split)
        out[split] = validate_split(split, scenarios, truths, roles)
    return out


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
