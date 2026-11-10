"""Command-line interface: `wmw <command>` (or `python -m whose_memory_wins <command>`).

The CLI is complete on its own; the dashboard only reads what the CLI produces.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from . import env  # noqa: F401  (environment setup first)
from .config import credential_status, load_config, mode_of, outputs_dir, validate_config


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cfg(args) -> dict:
    return load_config(Path(args.config) if args.config else None, mode=getattr(args, "mode", None))


def _resolve_run(cfg: dict, run: str) -> Path:
    p = Path(run)
    if p.exists():
        return p
    for root in (outputs_dir(cfg) / "runs", outputs_dir(cfg) / "diagnostics"):
        hits = list(root.glob(f"**/{run}")) if root.exists() else []
        hits = [h for h in hits if (h / "manifest.json").exists()]
        if hits:
            return hits[0]
    raise SystemExit(f"run not found: {run}")


def _guard_real(cfg: dict, args) -> None:
    if mode_of(cfg) != "real":
        return
    if not getattr(args, "yes_real", False):
        raise SystemExit("This command calls a paid API (REAL mode). Re-run with --yes-real to confirm. "
                         "Limits from the config apply: " + json.dumps(cfg["limits"]))
    cs = credential_status()
    if cs["ANTHROPIC_API_KEY"] != "set" and cs["ANTHROPIC_AUTH_TOKEN"] != "set":
        raise SystemExit("No Anthropic credential found. Set ANTHROPIC_API_KEY in .env (see .env.example). "
                         "Do not paste keys into chat.")


# --- commands ---------------------------------------------------------------------------
def cmd_check_config(args) -> None:
    cfg = _cfg(args)
    errs = validate_config(cfg)
    from .experiments.usage import describe_limits
    out = {"mode": mode_of(cfg), "decision_model": cfg["decision_model"]["model"],
           "config_errors": errs, "credentials": credential_status(),
           "limits_enforced_for_real_runs": describe_limits(cfg["limits"], cfg["pricing"])}
    if args.ping:
        if mode_of(cfg) != "real":
            out["ping"] = "skipped (mock mode)"
        else:
            from .agent.providers import make_provider
            p = make_provider({**cfg["decision_model"], "max_tokens": 256})
            try:
                r = p.create("Reply with the single word OK.", [{"role": "user", "content": "Ping."}], [])
                out["ping"] = {"ok": True, "model": r.model, "stop_reason": r.stop_reason,
                               "input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
            except Exception as exc:  # report, never hide
                out["ping"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    _print(out)
    if errs:
        sys.exit(1)


def cmd_dataset_validate(args) -> None:
    from .dataset import load_split, split_hash, validate_all
    res = validate_all()
    summary = {}
    for split, errs in res.items():
        scen, _ = load_split(split)
        summary[split] = {"scenarios": len(scen), "hash": split_hash(split), "errors": errs}
    _print(summary)
    if any(res.values()):
        sys.exit(1)


def cmd_dataset_freeze(args) -> None:
    from .agent.context import prompt_hashes
    from .dataset import DATASETS, SPLITS, split_hash, validate_all
    errs = validate_all()
    if any(errs.values()):
        raise SystemExit(f"refusing to freeze an invalid dataset: {errs}")
    path = DATASETS / "FROZEN.json"
    if path.exists() and not args.force:
        raise SystemExit(f"already frozen: {path.read_text()}")
    data = {"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "splits": {s: split_hash(s) for s in SPLITS}, "prompt_hashes": prompt_hashes(),
            "evaluation_tuned_after_freeze": False,
            "note": "Evaluation cases must not be used to tune prompts/code after this point. If they are, set "
                    "evaluation_tuned_after_freeze=true and record it in docs/DEVIATIONS_AND_LIMITATIONS.md."}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _print(data)


def cmd_fixtures(args) -> None:
    from .memory.fixtures import build_controlled_fixture
    cfg = _cfg(args)
    fdir = build_controlled_fixture(args.split, outputs_dir(cfg), cfg["retrieval"], rebuild=args.rebuild)
    print(fdir)


def _fixture_for(cfg: dict, split: str) -> Path:
    from .memory.fixtures import build_controlled_fixture
    return build_controlled_fixture(split, outputs_dir(cfg), cfg["retrieval"])


def cmd_run_create(args) -> None:
    from .experiments.runner import create_run
    cfg = _cfg(args)
    fdir = _fixture_for(cfg, args.split)
    run_dir = create_run(cfg, name=args.name, split=args.split, experiment_kind="core",
                         snapshot_dir=fdir / "retrieval", register_path=fdir / "source_register.sqlite",
                         scenario_ids=_csv(args.scenarios), conditions=_csv(args.conditions),
                         configs=_csv(args.configs), reps=args.reps, seed=args.seed)
    m = json.loads((run_dir / "manifest.json").read_text())
    print(f"Created run {m['run_id']} ({m['mode'].upper()}) with {m['n_trials']} planned trials at {run_dir}")
    print(f"Frozen-dataset check: {m['frozen_check']}")


def cmd_run_execute(args) -> None:
    from .experiments.runner import execute_run, load_manifest
    cfg = _cfg(args)
    run_dir = _resolve_run(cfg, args.run)
    m = load_manifest(run_dir)
    _guard_real(m["config"], args)
    if m["mode"] == "real":
        errs = validate_config(m["config"])
        if errs:
            raise SystemExit(f"refusing to execute: the run's config is invalid: {errs}")
    overrides = {k: v for k, v in {"max_api_calls": args.max_api_calls}.items() if v is not None}
    st = execute_run(run_dir, retry_failed=args.retry_failed, max_trials=args.max_trials,
                     limit_overrides=overrides or None)
    _print(st)


def cmd_run_status(args) -> None:
    from .experiments.runner import run_status
    _print(run_status(_resolve_run(_cfg(args), args.run)))


def cmd_run_list(args) -> None:
    cfg = _cfg(args)
    from .experiments.runner import run_status
    for root in (outputs_dir(cfg) / "runs", outputs_dir(cfg) / "diagnostics"):
        for man in sorted(root.glob("**/manifest.json")) if root.exists() else []:
            st = run_status(man.parent)
            print(f"{st['run_id']:<60} {st['kind']:<20} finished {st['finished']}/{st['planned']}")


def analysis_brief(res: dict, out: Path) -> dict:
    """CLI summary. Uses the same table rows as the dashboard and results_table.csv."""
    return {"run_id": res["run_id"], "mode": res["mode"], "provisional": res["provisional"],
            "scoring_version": res["scoring_version"], "scoring_warning": res["scoring_warning"],
            "progress": res["progress"], "by_config": {r["config"]: r for r in res["results_table"]},
            "prohibited_attempts_by_claim_status": {c: g["prohibited_attempts_by_claim_status"]
                                                    for c, g in res["by_config"].items()},
            "admission_vs_trace_conflicts": len(res["admission_vs_trace_conflicts"]),
            "comparisons": res["comparisons"], "output_dir": str(out)}


def cmd_analyze(args) -> None:
    from .analysis.report import write_report
    cfg = _cfg(args)
    run_dir = _resolve_run(cfg, args.run)
    out = outputs_dir(cfg) / "analysis" / run_dir.name
    res = write_report(run_dir, out)
    _print(analysis_brief(res, out))


def cmd_ingest_run(args) -> None:
    from .experiments.ingestion import admission_summary, create_decision_run_for_study, run_ingestion_study
    cfg = _cfg(args)
    _guard_real(cfg, args)
    if mode_of(cfg) == "real":
        print(f"Mem0 ingestion: hard ceiling of {cfg['limits']['max_api_calls']} extraction-LLM calls for the "
              "whole study. Mem0 does not report tokens, so NO token or dollar limit applies to ingestion.")
    sdir = run_ingestion_study(cfg, name=args.name, scenario_ids=_csv(args.scenarios),
                               conditions=_csv(args.conditions))
    print(f"Ingestion study written to {sdir}")
    _print(admission_summary(sdir))
    try:
        run_dir = create_decision_run_for_study(cfg, sdir, name=f"ingest-{args.name}", reps=args.reps,
                                                seed=args.seed, configs=_csv(args.configs))
    except ValueError as exc:
        raise SystemExit(str(exc))
    n = json.loads((run_dir / "manifest.json").read_text())["n_trials"]
    print(f"Created decision run {run_dir.name} ({n} trials) for the successful ingestion cases only. "
          "Execute it with:\n"
          f"  wmw run execute --run {run_dir.name}" + (" --yes-real" if mode_of(cfg) == "real" else ""))


def cmd_ingest_review(args) -> None:
    from .experiments.ingestion import admission_summary, set_admission_label
    sdir = Path(args.study)
    if args.case:
        set_admission_label(sdir, args.case, args.label, args.note or "")
    _print(admission_summary(sdir))


def cmd_diagnose(args) -> None:
    from .experiments.diagnostics import create_removal_run
    cfg = _cfg(args)
    run_dir = create_removal_run(_resolve_run(cfg, args.run), args.scenario, args.condition, _csv(args.remove),
                                 configs=_csv(args.configs), reps=args.reps)
    print(f"Created diagnostic run {run_dir.name} at {run_dir}. Execute with: wmw run execute --run {run_dir}")


def cmd_demo(args) -> None:
    """Offline MOCK demonstration: fixtures -> mock run -> analysis -> mock ingestion. No API calls."""
    from .analysis.report import write_report
    from .experiments.ingestion import run_ingestion_study
    from .experiments.runner import create_run, execute_run
    cfg = load_config(mode="mock")
    cfg["decision_model"]["mock_policy"] = args.policy
    fdir = _fixture_for(cfg, args.split)
    run_dir = create_run(cfg, name=f"demo-{args.policy}", split=args.split, experiment_kind="core",
                         snapshot_dir=fdir / "retrieval", register_path=fdir / "source_register.sqlite",
                         reps=args.reps, seed=args.seed)
    st = execute_run(run_dir, log=lambda s: None)
    print(f"MOCK run {st['run_id']}: finished {st['finished']}/{st['planned']}")
    res = write_report(run_dir, outputs_dir(cfg) / "analysis" / run_dir.name)
    for c, g in res["by_config"].items():
        print(f"  [MOCK] config {c}: unauthorized {g['unauthorized_attempt_rate']['display']}, "
              f"authorized success {g['authorized_exception_success']['display']}, "
              f"clean success {g['clean_success']['display']}")
    if not args.skip_ingestion:
        sdir = run_ingestion_study(cfg, name="demo", log=lambda s: None)
        print(f"MOCK ingestion study: {sdir}")
    print("These numbers come from a rule-based MOCK model and are NOT research findings.")


def cmd_dashboard(args) -> None:
    app = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"], check=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wmw", description="Whose Memory Wins research harness")
    ap.add_argument("--config", help="path to a YAML config (default configs/default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-config", help="validate config and show which credentials are set")
    p.add_argument("--mode", choices=["mock", "real"])
    p.add_argument("--ping", action="store_true", help="REAL mode: make one tiny API call to verify access")
    p.set_defaults(func=cmd_check_config)

    p = sub.add_parser("dataset", help="dataset validation / freezing")
    dsub = p.add_subparsers(dest="dcmd", required=True)
    q = dsub.add_parser("validate"); q.set_defaults(func=cmd_dataset_validate)
    q = dsub.add_parser("freeze"); q.add_argument("--force", action="store_true"); q.set_defaults(func=cmd_dataset_freeze)

    p = sub.add_parser("fixtures", help="build Track 1 controlled Mem0 fixtures + retrieval snapshots (offline)")
    p.add_argument("--split", default="evaluation", choices=["development", "evaluation"])
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_fixtures)

    p = sub.add_parser("run", help="create / execute / inspect decision runs")
    rsub = p.add_subparsers(dest="rcmd", required=True)
    q = rsub.add_parser("create", help="write a manifest (no model calls)")
    q.add_argument("--mode", choices=["mock", "real"], required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--split", default="evaluation", choices=["development", "evaluation"])
    q.add_argument("--scenarios"); q.add_argument("--conditions"); q.add_argument("--configs")
    q.add_argument("--reps", type=int, default=3); q.add_argument("--seed", type=int, default=20260101)
    q.set_defaults(func=cmd_run_create)
    q = rsub.add_parser("execute", help="execute or resume a run (saves after every trial)")
    q.add_argument("--run", required=True)
    q.add_argument("--yes-real", action="store_true", help="confirm paid API use for REAL runs")
    q.add_argument("--max-trials", type=int, help="stop after this many trials in this session")
    q.add_argument("--max-api-calls", type=int, help="override the cumulative API-call ceiling for this run")
    q.add_argument("--retry-failed", action="store_true", help="add a new attempt for trials whose latest is api_error")
    q.set_defaults(func=cmd_run_execute)
    q = rsub.add_parser("status"); q.add_argument("--run", required=True); q.set_defaults(func=cmd_run_status)
    q = rsub.add_parser("list"); q.set_defaults(func=cmd_run_list)

    p = sub.add_parser("analyze", help="metrics, tables and charts for a run")
    p.add_argument("--run", required=True); p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ingest", help="Track 2 normal-ingestion study")
    isub = p.add_subparsers(dest="icmd", required=True)
    q = isub.add_parser("run")
    q.add_argument("--mode", choices=["mock", "real"], required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--scenarios"); q.add_argument("--conditions")
    q.add_argument("--configs", help="comma list of configurations for the decision run (default A,B,C)")
    q.add_argument("--reps", type=int, default=1); q.add_argument("--seed", type=int, default=20260102)
    q.add_argument("--yes-real", action="store_true")
    q.set_defaults(func=cmd_ingest_run)
    q = isub.add_parser("review", help="show admission status or set a manual admission label")
    q.add_argument("--study", required=True); q.add_argument("--case")
    q.add_argument("--label", choices=["retained", "not_retained", "partially_retained"])
    q.add_argument("--note"); q.set_defaults(func=cmd_ingest_review)

    p = sub.add_parser("diagnose", help="memory-removal rerun (separate diagnostic run)")
    p.add_argument("--run", required=True); p.add_argument("--scenario", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--remove", required=True, help="comma list of M<rank> labels or kinds (claim,policy,...)")
    p.add_argument("--configs"); p.add_argument("--reps", type=int, default=1)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("demo", help="offline MOCK end-to-end demonstration")
    p.add_argument("--policy", default="cautious",
                   choices=["gullible", "cautious", "self_correcting", "malformed", "flaky"])
    p.add_argument("--split", default="development", choices=["development", "evaluation"])
    p.add_argument("--reps", type=int, default=1); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-ingestion", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("dashboard", help="open the Streamlit research dashboard")
    p.set_defaults(func=cmd_dashboard)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
