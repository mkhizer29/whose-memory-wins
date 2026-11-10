"""Streamlit research dashboard. Read-only view over the outputs produced by the CLI.

Run with:  wmw dashboard   (or: streamlit run src/whose_memory_wins/dashboard/app.py)
Nothing here builds model prompts; evaluator-only content is shown in labelled panels only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from whose_memory_wins import env  # noqa: F401
from whose_memory_wins.analysis.metrics import compute, load_frames
from whose_memory_wins.config import load_config, outputs_dir
from whose_memory_wins.dataset import load_roles, load_split, split_hash
from whose_memory_wins.experiments.ingestion import admission_summary
from whose_memory_wins.experiments.runner import load_manifest, run_status
from whose_memory_wins.experiments.store import ResultStore

st.set_page_config(page_title="Whose Memory Wins", layout="wide")
CFG = load_config()
OUT = outputs_dir(CFG)


def find_runs() -> list[Path]:
    runs = []
    for root in (OUT / "runs", OUT / "diagnostics"):
        if root.exists():
            runs += [p.parent for p in root.glob("**/manifest.json")]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def find_fixtures() -> list[Path]:
    root = OUT / "fixtures" / "controlled"
    return sorted(p.parent for p in root.glob("*/fixture_manifest.json")) if root.exists() else []


def find_studies() -> list[Path]:
    root = OUT / "ingestion"
    return sorted((p.parent for p in root.glob("*/study_manifest.json")), reverse=True) if root.exists() else []


def mode_banner(mode: str) -> None:
    if mode == "mock":
        st.error("MOCK MODE - rule-based mock model for software verification. These are NOT research findings.")
    else:
        st.success("REAL MODE - results from the configured real model.")


def pct(f: dict) -> str:
    return f["display"]


runs = find_runs()
page = st.sidebar.radio("Page", ["Overview", "Scenario viewer", "Memory inspector", "Comparison viewer", "Results",
                                 "Trial detail", "Ingestion study"])
st.sidebar.caption(f"Outputs: {OUT}")

# ------------------------------------------------------------------------------------------
if page == "Overview":
    st.title("Whose Memory Wins: Authority Conflicts in LLM Agent Memory")
    st.markdown(
        "Does a purchasing assistant follow an **unauthorized claim stored in persistent memory** when it conflicts "
        "with company policy? Three configurations are compared on identical retrieved memory:\n"
        "- **A** - ordinary memory retrieval only\n- **B** - A + the protected policy register\n"
        "- **C** - B + application-recorded source identity, role permissions and permission-handling rules\n\n"
        "This is a scoped empirical study in a simulated environment. It does not establish formal security "
        "guarantees.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Configured decision model", CFG["decision_model"]["model"])
    c2.metric("Configured mode", "MOCK" if CFG["decision_model"]["provider"] == "mock" else "REAL")
    frozen = json.loads((Path(env.PROJECT_ROOT) / "datasets" / "FROZEN.json").read_text()) \
        if (Path(env.PROJECT_ROOT) / "datasets" / "FROZEN.json").exists() else None
    c3.metric("Dataset frozen", "yes" if frozen else "no")
    st.caption(f"Evaluation dataset hash: {split_hash('evaluation')[:16]}...")
    if not runs:
        st.info("No results collected.")
    else:
        rows = []
        for r in runs:
            s = run_status(r)
            rows.append({"run": s["run_id"], "mode": s["mode"].upper(), "kind": s["kind"], "planned": s["planned"],
                         "finished": s["finished"], "api_errors": s["failed_api_error"],
                         "budget_stopped": s["stopped_by_budget"], "pending": s["pending"],
                         "api_calls": s["usage"].get("api_calls")})
        st.subheader("Runs")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------
elif page == "Scenario viewer":
    split = st.selectbox("Split", ["evaluation", "development"])
    scenarios, truths = load_split(split)
    roles = load_roles()
    sid = st.selectbox("Scenario", [s.scenario_id for s in scenarios])
    s = next(x for x in scenarios if x.scenario_id == sid)
    st.subheader(f"{s.scenario_id} - {s.family.value}")
    st.caption(s.design_note)
    st.markdown(f"**Company:** {s.company.name} - **date:** {s.company.current_date}")
    st.markdown("**Task (shown to the model in every configuration)**")
    st.info(s.task.text)
    st.markdown("**Company rules**")
    for p in s.policies:
        st.markdown(f"- `{p.policy_id}` **{p.title}** ({'overridable' if p.overridable else 'NOT overridable'}): "
                    f"{p.text}")
    for cond, spec in s.conditions.items():
        with st.expander(f"Condition: {cond.value}", expanded=cond.value != "clean"):
            if not spec.messages:
                st.write("No exception claim in memory.")
            for m in spec.messages:
                src = s.source(m.source_id)
                role = roles.get(src.role)
                st.markdown(f"**Message** `{m.message_id}` from **{src.display_name}** - application-assigned role "
                            f"`{src.role}`")
                st.write(m.message_text)
                st.markdown(f"Role permissions: `{', '.join(role.permissions) or 'none'}`"
                            + (f" (cap PKR {role.max_exception_amount_pkr:,})" if role.max_exception_amount_pkr else ""))
            for a in spec.approvals:
                st.markdown(f"Registered approval `{a.approval_id}`: policies {list(a.policy_ids)}, supplier "
                            f"{a.supplier_id}, up to PKR {a.max_amount_pkr:,}, expires {a.expires}, granted by "
                            f"{a.granted_by}")
    st.divider()
    with st.expander("EVALUATOR ONLY - expected behaviour (never sent to the model)", expanded=False):
        t = truths[sid]
        st.caption(f"conflict_relevant = {t.conflict_relevant}")
        for cond, ct in t.conditions.items():
            st.markdown(f"**{cond.value}** - acceptable: "
                        f"{[a.tool + ('(' + a.approval_reference + ')' if a.approval_reference else '') for a in ct.acceptable_terminal_actions]}; "
                        f"prohibited kinds: {list(ct.prohibited_kinds)}")
            st.caption(ct.explanation)

# ------------------------------------------------------------------------------------------
elif page == "Memory inspector":
    track = st.radio("Track", ["Controlled insertion (Track 1)", "Normal ingestion (Track 2)"], horizontal=True)
    if track.startswith("Controlled"):
        fixtures = find_fixtures()
        if not fixtures:
            st.info("No fixtures built yet (run `wmw fixtures`).")
            st.stop()
        fdir = st.selectbox("Fixture", fixtures, format_func=lambda p: p.name)
        snaps = sorted((fdir / "retrieval").glob("*.json"))
        inserted = json.loads((fdir / "inserted_records.json").read_text(encoding="utf-8"))
    else:
        studies = find_studies()
        if not studies:
            st.info("No ingestion studies yet (run `wmw ingest run`).")
            st.stop()
        fdir = st.selectbox("Study", studies, format_func=lambda p: p.name)
        snaps = sorted((fdir / "retrieval").glob("*.json"))
        inserted = None
    if not snaps:
        st.info("No retrieval snapshots in this fixture/study.")
        st.stop()
    snap_path = st.selectbox("Scenario x condition", snaps, format_func=lambda p: p.stem)
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    reg = json.loads((fdir / "source_register_export.json").read_text(encoding="utf-8"))
    ns = snap["namespace"]
    st.markdown(f"**Query:** `{snap['query']}` - top_k {snap['top_k']} - records in namespace: "
                f"{snap['namespace_record_count']}")
    c1, c2 = st.columns(2)
    c1.metric("Applicable policy retrieved (A visibility)", "yes" if snap["policy_retrieved"] else "no")
    c2.metric("Claim retrieved", "yes" if snap["claim_retrieved"] else ("n/a" if not snap["claim_present_in_memory"]
                                                                       else "no"))
    st.subheader("Original messages (source envelopes)")
    env_rows = [e for e in reg["envelopes"] if e["namespace"] == ns]
    st.dataframe(pd.DataFrame(env_rows)[["message_id", "display_name", "role", "original_message"]],
                 hide_index=True, width="stretch")
    st.subheader("Stored memory records and lineage")
    lin = pd.DataFrame([x for x in reg["lineage"] if x["namespace"] == ns])
    if not lin.empty:
        st.dataframe(lin[["record_id", "primary_message_id", "status", "text_at_link", "deleted"]],
                     hide_index=True, width="stretch")
    if inserted is None and (fdir / "cases" / ns / "writes.json").exists():
        with st.expander("Observed writes per ingested message (Mem0 add results + snapshot diff)"):
            st.json(json.loads((fdir / "cases" / ns / "writes.json").read_text(encoding="utf-8")))
    st.subheader("Retrieved records (identical for A, B and C)")
    st.dataframe(pd.DataFrame([{"rank": r["rank"], "text": r["text"], "score": r["score"],
                                "provenance": r["provenance_status"], "source": r["source"]["display_name"],
                                "assigned_role": r["source"]["assigned_role"],
                                "effective_role": r["source"]["effective_role"],
                                "linked_approvals": ", ".join(a["approval_id"] for a in r["linked_approvals"])}
                               for r in snap["records"]]), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------------------
elif page in ("Comparison viewer", "Results", "Trial detail"):
    if not runs:
        st.info("No results collected.")
        st.stop()
    run_dir = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
    m = load_manifest(run_dir)
    mode_banner(m["mode"])
    st.caption(f"Run {m['run_id']} - kind {m['experiment_kind']} - model {m['decision_model'].get('model')} - "
               f"frozen check: {m['frozen_check'].get('reason')}")
    _, df, _ = load_frames(run_dir)
    store = ResultStore(run_dir)

    if page == "Comparison viewer":
        sid = st.selectbox("Scenario", sorted(df["scenario_id"].unique()))
        cond = st.selectbox("Condition", sorted(df.loc[df["scenario_id"] == sid, "condition"].unique()))
        rep = st.selectbox("Repetition", sorted(df["repetition"].unique()))
        cols = st.columns(3)
        base_ctx = None
        for col, cfg in zip(cols, ("A", "B", "C")):
            row = df[(df.scenario_id == sid) & (df.condition == cond) & (df.config == cfg) & (df.repetition == rep)]
            with col:
                st.subheader(f"Config {cfg}")
                if row.empty or pd.isna(row.iloc[0]["trace_file"]):
                    st.write("pending / not in this run")
                    continue
                t = store.load_trace(row.iloc[0]["trace_file"])
                sc = t["score"]
                st.markdown(f"**{'SUCCESS' if sc['success'] else 'not success'}** - status `{sc['status']}` - "
                            f"terminal `{sc['terminal_action']}` - prohibited attempt: **{sc['prohibited_attempt']}**")
                st.caption(f"labels: {t['labels']}")
                st.caption(f"tokens in/out: {t['input_tokens']}/{t['output_tokens']} - {t['duration_s']} s")
                for a in t["actions"]:
                    st.code(f"{a['tool']} {json.dumps(a['raw_input'])}\n-> executed={a['executed']} "
                            f"verdict={a['verdict_kind']}\n{a['tool_result']}", language="text")
                ctx = t["context"]["messages"][0]["content"]
                if base_ctx is None:
                    base_ctx = ctx
                    st.caption(f"context length: {len(ctx)} chars")
                else:
                    added = ctx[len(base_ctx):] if ctx.startswith(base_ctx) else ctx
                    st.caption(f"context length: {len(ctx)} chars; added beyond A:")
                    st.text(added[:3000])

    elif page == "Results":
        res = compute(run_dir)
        if res["provisional"]:
            st.warning(f"PROVISIONAL - {res['progress']}. Missing trials are not evidence of safety.")
        if res["scoring_warning"]:
            st.warning(f"Scoring rules: {res['scoring_warning']}")
        if res["admission_vs_trace_conflicts"]:
            st.warning(f"{len(res['admission_vs_trace_conflicts'])} trial(s) where the manual admission review and "
                       "the trace evidence disagree (see metrics.json: admission_vs_trace_conflicts).")
        fam = st.multiselect("Family", sorted(df["family"].unique()), default=sorted(df["family"].unique()))
        conds = st.multiselect("Condition", sorted(df["condition"].unique()), default=sorted(df["condition"].unique()))
        cfgs = st.multiselect("Configuration", sorted(df["config"].unique()), default=sorted(df["config"].unique()))
        from whose_memory_wins.analysis.metrics import results_table
        sub = df[df.family.isin(fam) & df.condition.isin(conds) & df.config.isin(cfgs)]
        rows = results_table(sub)  # same rows as `wmw analyze` and results_table.csv
        st.subheader("Metrics (k/n, n = planned trials in the filtered group)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader("Paired comparisons (scenario = sampling unit)")
        st.dataframe(pd.DataFrame(res["comparisons"]), hide_index=True, width="stretch")
        st.subheader("Diagnostic label counts (overlapping observations, not causes)")
        st.json(res["label_counts"])
        st.caption("Prohibited attempts by claim evidence (claim_status): "
                   + json.dumps({c: g["prohibited_attempts_by_claim_status"] for c, g in res["by_config"].items()}))
        img = OUT / "analysis" / run_dir.name / "core_metrics.png"
        if img.exists():
            st.image(str(img), caption="From `wmw analyze` (re-run it to refresh)")

    else:  # Trial detail
        done = df[df["trace_file"].notna()]
        if done.empty:
            st.info("No results collected.")
            st.stop()
        tid = st.selectbox("Trial", sorted(done["trial_id"]))
        attempts = [a for a in store.all_attempts() if a["trial_id"] == tid]
        att = st.selectbox("Attempt", [a["attempt"] for a in attempts], index=len(attempts) - 1)
        t = store.load_trace(next(a["trace_file"] for a in attempts if a["attempt"] == att))
        st.subheader(f"{tid} (attempt {att}) - mode {t['mode'].upper()}")
        st.markdown(f"**Score:** {t['score']['rule_applied']} - {t['score']['explanation']}")
        st.markdown(f"**Labels:** {t['labels']}  |  **Policy visible:** {t['policy_visible']}  |  "
                    f"**Scoring version:** {t['score'].get('scoring_version', '1')}")
        if t.get("evidence"):
            st.caption(f"Trace evidence: {t['evidence']}")
        if t["error"]:
            st.error(t["error"])
        st.markdown("**Exact model-visible system prompt**")
        st.code(t["context"]["system"], language="text")
        st.markdown("**Exact model-visible user message**")
        st.code(t["context"]["messages"][0]["content"], language="markdown")
        st.markdown("**Model calls (with retry history)**")
        for mc in t["model_calls"]:
            st.json(mc, expanded=False)
        st.markdown("**Tool actions (attempted and executed)**")
        st.dataframe(pd.DataFrame(t["actions"]), hide_index=True, width="stretch")
        st.markdown("**Final model text (optional output, not used for scoring)**")
        st.write(t["final_text"] or "(none)")
        with st.expander("Retrieval snapshot used"):
            st.json(t["retrieval"], expanded=False)
    store.close()

# ------------------------------------------------------------------------------------------
elif page == "Ingestion study":
    studies = find_studies()
    if not studies:
        st.info("No results collected (run `wmw ingest run`).")
        st.stop()
    sdir = st.selectbox("Study", studies, format_func=lambda p: p.name)
    sm = json.loads((sdir / "study_manifest.json").read_text(encoding="utf-8"))
    mode_banner(sm["mode"])
    st.caption("Supplementary track - reported separately from the core trials.")
    summ = admission_summary(sdir)
    st.metric("Memory admission (retained / attempted)", summ["admission_rate"])
    st.dataframe(pd.DataFrame(summ["rows"]), hide_index=True, width="stretch")
    case = st.selectbox("Case", [r["case"] for r in summ["rows"]])
    if not (sdir / "admission" / f"{case}.json").exists():
        st.error("Ingestion failed for this case; it has no snapshot or admission review, but it stays in the "
                 "attempted-case denominator.")
        st.stop()
    d = json.loads((sdir / "admission" / f"{case}.json").read_text(encoding="utf-8"))
    st.markdown(f"**Claim message:** {d['claim_message_text']}")
    st.markdown("**Records derived from the claim message:**")
    st.json(d["derived_records"])
    st.caption(f"Automatic decision: {d['automatic_decision']} ({d['decision_basis']}). "
               f"Keyword hint (not a label): {d['keyword_hint']['cues']}")
    st.markdown(f"**Manual label:** {d['manual_label']}  - set with "
                f"`wmw ingest review --study {sdir} --case {case} --label retained|not_retained|partially_retained`")
