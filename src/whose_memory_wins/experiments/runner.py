"""Experiment manifests and execution (create -> execute/resume -> status).

* A manifest fixes every setting (config, model, prompts, dataset hash, fixture, trial list,
  randomized order + seed) BEFORE any model call. Execution always uses the manifest's
  config, so a resumed run cannot silently change settings.
* Results are saved after every trial. Resume skips trials that already have a finished
  attempt. Failed attempts are kept; ``--retry-failed`` adds a new attempt, never overwrites.
* A local seed fixes the trial order only. Provider responses are not reproducible.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable

from ..agent.context import CONFIGS, build_context, prompt_hashes
from ..agent.loop import run_agent
from ..agent.providers import make_provider
from ..config import mode_of, outputs_dir
from ..dataset import DATASETS, load_roles, load_split, split_hash, stable_hash
from ..evaluation.labels import diagnostic_labels, trial_evidence
from ..evaluation.scorer import score_trial
from ..provenance.register import SourceRegister
from ..schemas.records import SCORING_VERSION, TrialResult
from ..schemas.scenario import Condition, Scenario
from ..snapshots import verify_snapshot_file
from ..tools.definitions import TOOL_DEFINITIONS
from ..tools.simulator import PurchasingSimulator
from .store import FAILED_STATUSES, FINISHED_STATUSES, INCOMPLETE_STATUSES, ResultStore
from .usage import UsageTracker, describe_limits

PACKAGES = ("mem0ai", "anthropic", "fastembed", "qdrant-client", "pydantic", "pandas", "matplotlib", "streamlit")


def trial_id(scenario_id: str, condition: str, config: str, rep: int) -> str:
    """Stable trial ID: ``<scenario>__<condition>__<config>__r<rep>``."""
    return f"{scenario_id}__{condition}__{config}__r{rep}"


def frozen_check(split: str) -> dict[str, Any]:
    """Whether the split and prompts still match ``datasets/FROZEN.json`` (held-out integrity)."""
    path = DATASETS / "FROZEN.json"
    if not path.exists():
        return {"frozen": False, "untouched": False, "reason": "dataset not frozen (run `wmw dataset freeze`)"}
    fr = json.loads(path.read_text(encoding="utf-8"))
    same_data = fr["splits"].get(split) == split_hash(split)
    same_prompts = fr["prompt_hashes"] == prompt_hashes()
    reasons = []
    if not same_data:
        reasons.append("dataset changed since freeze")
    if not same_prompts:
        reasons.append("prompts/tools changed since freeze")
    if fr.get("evaluation_tuned_after_freeze"):
        reasons.append("evaluation cases were used for tuning after freezing (see docs/DEVIATIONS_AND_LIMITATIONS.md)")
    return {"frozen": True, "frozen_at": fr["frozen_at"], "untouched": not reasons,
            "reason": "; ".join(reasons) or "matches frozen dataset and prompt hashes"}


def environment_info() -> dict[str, Any]:
    """Python, platform and key package versions, recorded in every manifest."""
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": pk}


def create_run(cfg: dict[str, Any], *, name: str, split: str, experiment_kind: str, snapshot_dir: Path,
               register_path: Path, scenario_ids: list[str] | None = None, conditions: list[str] | None = None,
               configs: list[str] | None = None, reps: int = 3, seed: int = 20260101,
               extra: dict[str, Any] | None = None, run_root: Path | None = None,
               pairs: list[tuple[str, str]] | None = None) -> Path:
    """Write a run manifest. No model calls.

    ``pairs``: exact (scenario_id, condition) cases to schedule, e.g. the successful cases of an
    ingestion study. When given, ``scenario_ids``/``conditions`` must be omitted, and no other
    combination of those scenarios and conditions is scheduled.
    """
    scenarios, _ = load_split(split)
    if pairs is not None:
        if scenario_ids or conditions:
            raise ValueError("pass either exact pairs or scenario_ids/conditions, not both")
        pairs = [(str(a), str(b)) for a, b in pairs]
        if not pairs:
            raise ValueError("no (scenario, condition) pairs to schedule")
        known = {(s.scenario_id, c.value) for s in scenarios for c in s.conditions}
        unknown = sorted(set(pairs) - known)
        if unknown:
            raise ValueError(f"unknown (scenario, condition) pairs for split {split}: {unknown}")
        scenario_ids = sorted({sid for sid, _ in pairs})
    if scenario_ids:
        missing = set(scenario_ids) - {s.scenario_id for s in scenarios}
        if missing:
            raise ValueError(f"unknown scenarios for split {split}: {sorted(missing)}")
        scenarios = [s for s in scenarios if s.scenario_id in scenario_ids]
    configs = configs or list(CONFIGS)
    mode = mode_of(cfg)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{mode}-{name}-{stamp}"
    run_dir = (run_root or outputs_dir(cfg) / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trials = []
    for s in scenarios:
        conds = [c.value for c in s.conditions if conditions is None or c.value in conditions]
        if pairs is not None:
            conds = [c for c in conds if (s.scenario_id, c) in pairs]
        for cond in conds:
            snap_file = snapshot_dir / f"{s.scenario_id}__{cond}.json"
            if not snap_file.exists():
                raise FileNotFoundError(f"missing retrieval snapshot {snap_file}")
            # Recompute the hash from the actual content; refuse a snapshot whose content and stored hash disagree.
            snap_hash = verify_snapshot_file(snap_file)["snapshot_hash"]
            for config in configs:
                for rep in range(1, reps + 1):
                    trials.append({"trial_id": trial_id(s.scenario_id, cond, config, rep),
                                   "scenario_id": s.scenario_id, "family": s.family.value, "condition": cond,
                                   "config": config, "repetition": rep, "snapshot_file": str(snap_file),
                                   "snapshot_hash": snap_hash, "namespace": f"{s.scenario_id}__{cond}"})
    order = [t["trial_id"] for t in trials]
    random.Random(seed).shuffle(order)
    provider_settings = make_provider(cfg["decision_model"]).settings()
    manifest = {
        "run_id": run_id, "mode": mode, "experiment_kind": experiment_kind, "name": name, "split": split,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed, "seed_note": "The seed fixes trial order only. Provider responses are not reproducible "
                                   "even with identical inputs.",
        "config": cfg, "config_hash": stable_hash(cfg), "decision_model": provider_settings,
        "dataset_hash": split_hash(split), "frozen_check": frozen_check(split),
        "prompt_hashes": prompt_hashes(), "tool_definitions": TOOL_DEFINITIONS,
        "retrieval": cfg["retrieval"], "snapshot_dir": str(snapshot_dir), "register_path": str(register_path),
        "environment": environment_info(), "n_trials": len(trials), "trials": trials, "order": order,
        "configs": configs, "reps": reps, "conditions": conditions or "all",
        "pairs": [list(p) for p in pairs] if pairs is not None else "all selected",
        "scoring_version": SCORING_VERSION,
        "snapshot_hash_method": "sha256 of canonical JSON of the snapshot content excluding snapshot_hash "
                                "(whose_memory_wins.snapshots); re-verified before any model call",
    }
    if extra:
        manifest.update(extra)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return run_dir


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a run's manifest.json."""
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


def run_status(run_dir: Path) -> dict[str, Any]:
    """Progress of a run: finished / failed / budget-stopped / pending, plus usage so far."""
    m = load_manifest(run_dir)
    store = ResultStore(run_dir)
    latest = {r["trial_id"]: r for r in store.latest_attempts()}
    n_attempts = len(store.all_attempts())
    store.close()
    counts = {"planned": m["n_trials"], "finished": 0, "failed_api_error": 0, "stopped_by_budget": 0, "pending": 0}
    by_status: dict[str, int] = {}
    for t in m["trials"]:
        r = latest.get(t["trial_id"])
        if r is None:
            counts["pending"] += 1
            continue
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        if r["status"] in FINISHED_STATUSES:
            counts["finished"] += 1
        elif r["status"] in FAILED_STATUSES:
            counts["failed_api_error"] += 1
        elif r["status"] in INCOMPLETE_STATUSES:
            counts["stopped_by_budget"] += 1
    usage_path = Path(run_dir) / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}
    return {"run_id": m["run_id"], "mode": m["mode"], "kind": m["experiment_kind"],
            "scoring_version": m.get("scoring_version", "1"), **counts,
            "finished_by_status": by_status, "total_attempts_recorded": n_attempts, "usage": usage,
            "complete": counts["finished"] == m["n_trials"]}


def _scenario_index(split: str) -> tuple[dict[str, Scenario], dict]:
    scenarios, truths = load_split(split)
    return {s.scenario_id: s for s in scenarios}, truths


def execute_trial(spec: dict[str, Any], manifest: dict[str, Any], scenario: Scenario, truth, roles,
                  register: SourceRegister, provider, attempt: int, usage: UsageTracker,
                  sleep: Callable[[float], None] = time.sleep) -> TrialResult:
    """Run, score and label one trial attempt (does not save it).

    The snapshot is re-verified against the manifest hash before the provider is used.
    Scoring uses only the action trace; labels and evidence are observational.
    """
    cfg = manifest["config"]
    snapshot = verify_snapshot_file(Path(spec["snapshot_file"]), spec["snapshot_hash"])
    approvals = register.approvals(spec["namespace"])
    context = build_context(spec["config"], scenario, snapshot, roles)
    sim = PurchasingSimulator(scenario, approvals, roles)
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()
    loop = run_agent(provider, context, sim, max_model_calls=cfg["agent"]["max_model_calls"],
                     max_tool_calls=cfg["agent"]["max_tool_calls"], max_attempts=cfg["retries"]["max_attempts"],
                     backoff_s=cfg["retries"]["backoff_s"], before_call=usage.before_call,
                     after_call=usage.after_call, sleep=sleep)
    cond_truth = truth.conditions[Condition(spec["condition"])]
    score = score_trial(cond_truth, sim.actions, sim.terminal, loop.status)
    policy_visible = spec["config"] in ("B", "C") or bool(snapshot["policy_retrieved"])
    visible_text = "\n".join([context["system"], *(str(m["content"]) for m in context["messages"])])
    evidence = trial_evidence(scenario, spec["condition"], score, sim.actions, snapshot, visible_text)
    labels = diagnostic_labels(spec["condition"], score, policy_visible, snapshot, evidence)
    in_tok = sum(c.input_tokens for c in loop.model_calls)
    out_tok = sum(c.output_tokens for c in loop.model_calls)
    return TrialResult(
        trial_id=spec["trial_id"], run_id=manifest["run_id"], mode=manifest["mode"],
        experiment_kind=manifest["experiment_kind"], scenario_id=spec["scenario_id"], family=spec["family"],
        condition=spec["condition"], config=spec["config"], repetition=spec["repetition"], attempt=attempt,
        model=provider.model, started_at=started.isoformat(),
        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), duration_s=round(time.perf_counter() - t0, 3),
        context={"system": context["system"], "messages": context["messages"],
                 "tools_hash": prompt_hashes()["tool_definitions"]},
        retrieval={"snapshot_file": spec["snapshot_file"], "snapshot_hash": spec["snapshot_hash"],
                   "query": snapshot["query"], "records": snapshot["records"],
                   "claim_retrieved": snapshot["claim_retrieved"], "policy_retrieved": snapshot["policy_retrieved"],
                   "registered_approval_ids": snapshot["registered_approval_ids"]},
        policy_visible=policy_visible, model_calls=loop.model_calls, actions=sim.actions,
        final_text=loop.final_text, score=score, labels=labels, evidence=evidence, input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=usage.cost_usd(in_tok, out_tok), error=loop.error)


def execute_run(run_dir: Path, *, retry_failed: bool = False, max_trials: int | None = None,
                limit_overrides: dict[str, int] | None = None, provider=None, log=print,
                sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Execute or resume a run, saving each trial as soon as it finishes.

    Before any model call it checks the scoring version, every snapshot hash, the
    dataset hash and the prompt hashes; any mismatch stops the run. Real runs also
    check the usage ceilings before every trial and every request.
    """
    run_dir = Path(run_dir)
    m = load_manifest(run_dir)
    cfg = m["config"]
    if m.get("scoring_version") != SCORING_VERSION:
        raise RuntimeError(f"run {m['run_id']} was created under scoring version {m.get('scoring_version', '1')}; "
                           f"the harness now scores with version {SCORING_VERSION}. Results scored under different "
                           "rules are never mixed in one run. Create a new run (the old one is kept unchanged).")
    # Verify every frozen retrieval snapshot of this run BEFORE any model call (also on resume).
    for snap_file, snap_hash in sorted({(t["snapshot_file"], t["snapshot_hash"]) for t in m["trials"]}):
        verify_snapshot_file(Path(snap_file), snap_hash)
    limits = {**cfg["limits"], **(limit_overrides or {})}
    if m["mode"] == "mock":  # mock calls cost nothing; ceilings exist to protect real spend
        limits = {k: 10 ** 12 for k in limits}
    if limit_overrides:
        with open(run_dir / "session_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "limit_overrides": limit_overrides}) + "\n")
    provider = provider or make_provider(cfg["decision_model"])
    if provider.mode != m["mode"]:
        raise RuntimeError(f"provider mode {provider.mode} does not match run mode {m['mode']}; "
                           "mock and real results are never mixed in one run")
    scen, truths = _scenario_index(m["split"])
    if split_hash(m["split"]) != m["dataset_hash"]:
        raise RuntimeError("dataset changed since this run's manifest was created; create a new run")
    if prompt_hashes() != m["prompt_hashes"]:
        raise RuntimeError("prompts/tools changed since this run's manifest was created; create a new run")
    roles = load_roles()
    register = SourceRegister(Path(m["register_path"]), roles)
    store = ResultStore(run_dir)
    usage = UsageTracker(run_dir / "usage.json", limits, cfg["pricing"])
    worst = cfg["agent"]["max_model_calls"] * cfg["retries"]["max_attempts"]
    specs = {t["trial_id"]: t for t in m["trials"]}
    done = 0
    stop_reason = None
    try:
        for tid in m["order"]:
            latest = store.latest(tid)
            if latest is not None:
                if latest["status"] in FINISHED_STATUSES:
                    continue
                if latest["status"] in FAILED_STATUSES and not retry_failed:
                    continue
            if max_trials is not None and done >= max_trials:
                stop_reason = f"--max-trials {max_trials} reached"
                break
            if m["mode"] == "real":
                reason = usage.can_start_trial(worst)
                if reason:
                    stop_reason = reason
                    break
            spec = specs[tid]
            attempt = store.next_attempt(tid)
            result = execute_trial(spec, m, scen[spec["scenario_id"]], truths[spec["scenario_id"]], roles,
                                   register, provider, attempt, usage, sleep=sleep)
            store.save(result)
            done += 1
            s = result.score
            log(f"[{m['mode'].upper()}] {tid} a{attempt}: {s.status} terminal={s.terminal_action} "
                f"success={s.success} prohibited={s.prohibited_attempt}")
            if s.status == "budget_stop":
                stop_reason = result.error
                break
    finally:
        store.close()
        register.close()
    status = run_status(run_dir)
    status["limits_enforced"] = describe_limits(limits, cfg["pricing"])
    status["stopped_because"] = stop_reason
    status["executed_this_session"] = done
    return status
