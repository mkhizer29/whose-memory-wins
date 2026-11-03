"""Identical base retrieval across A/B/C; no expected-answer or hidden-permission leakage."""
from __future__ import annotations

import json

import pytest

from whose_memory_wins.agent.context import build_context
from whose_memory_wins.dataset import load_split
from whose_memory_wins.memory.fixtures import load_snapshot, namespace, namespace_store_dir
from whose_memory_wins.memory.mem0_store import Mem0Store

pytestmark = pytest.mark.embeddings


def _memory_block(text: str) -> str:
    return text.split("## Records retrieved from long-term memory", 1)[1].split("\n## ", 1)[0].strip()


def test_abc_share_identical_retrieval_and_extend_each_other(eval_fixture, roles):
    scenarios, _ = load_split("evaluation")
    for s in scenarios:
        for cond in s.conditions:
            snap = load_snapshot(eval_fixture, s.scenario_id, cond.value)
            a, b, c = (build_context(k, s, snap, roles) for k in "ABC")
            ua, ub, uc = (x["messages"][0]["content"] for x in (a, b, c))
            assert _memory_block(ua) == _memory_block(ub) == _memory_block(uc)
            assert ub.startswith(ua) and uc.startswith(ub)  # B and C only ADD information
            assert a["system"] == b["system"] == c["system"]
            assert a["tools"] == b["tools"] == c["tools"]


def test_a_and_b_prompts_identical_for_untrusted_vs_authorized(eval_fixture, roles):
    """If A/B prompts are byte-identical across the two conflict conditions, no hidden channel can leak
    the source, the approval register, or the expected answer into A or B."""
    scenarios, _ = load_split("evaluation")
    for s in scenarios:
        su = load_snapshot(eval_fixture, s.scenario_id, "untrusted_claim")
        sa = load_snapshot(eval_fixture, s.scenario_id, "authorized_exception")
        for k in "AB":
            pu = build_context(k, s, su, roles)["messages"][0]["content"]
            pa = build_context(k, s, sa, roles)["messages"][0]["content"]
            assert pu == pa, f"{s.scenario_id} config {k} differs between conditions"
        cu = build_context("C", s, su, roles)["messages"][0]["content"]
        ca = build_context("C", s, sa, roles)["messages"][0]["content"]
        assert cu != ca  # C is supposed to differ: that is its intervention


def test_no_ground_truth_or_register_details_in_prompts(eval_fixture, roles):
    scenarios, truths = load_split("evaluation")
    for s in scenarios:
        t = truths[s.scenario_id]
        forbidden_everywhere = ["acceptable_terminal_actions", "prohibited_kinds", "conflict_relevant",
                                "unnecessary_approval_is_failure", s.design_note]
        for ct in t.conditions.values():
            forbidden_everywhere.append(ct.explanation)
        for cond in s.conditions:
            snap = load_snapshot(eval_fixture, s.scenario_id, cond.value)
            for k in "ABC":
                ctx = build_context(k, s, snap, roles)
                blob = json.dumps(ctx)
                for f in forbidden_everywhere:
                    assert f not in blob, f"{s.scenario_id}/{cond.value}/{k} leaks {f[:40]!r}"
                if k in "AB":
                    for f in ("authorized_approver", "policy_admin", "supplier)", "assigned role", "provenance",
                              "registered authorization", "SRC-", "wmw_"):
                        assert f not in ctx["messages"][0]["content"], f"{k} leaks {f!r}"


def _records(fdir, cfg, ns):
    store = Mem0Store(namespace_store_dir(fdir, ns), cfg["retrieval"])
    try:
        return store.records(ns)
    finally:
        store.close()


def test_fixture_namespaces_are_isolated(eval_fixture, mock_cfg):
    scenarios, _ = load_split("evaluation")
    s = scenarios[0]
    clean = _records(eval_fixture, mock_cfg, namespace(s.scenario_id, "clean"))
    unt = _records(eval_fixture, mock_cfg, namespace(s.scenario_id, "untrusted_claim"))
    assert not set(clean) & set(unt)
    assert len(unt) == len(clean) + 1  # exactly the claim record added
    assert not any("pre-approved" in r["memory"] for r in clean.values())


def test_retrieval_snapshots_match_live_search(eval_fixture, mock_cfg):
    """The frozen snapshot equals what Mem0 returns now (deterministic local embeddings)."""
    scenarios, _ = load_split("evaluation")
    for s in scenarios[:6]:
        for cond in ("untrusted_claim", "authorized_exception"):
            snap = load_snapshot(eval_fixture, s.scenario_id, cond)
            store = Mem0Store(namespace_store_dir(eval_fixture, snap["namespace"]), mock_cfg["retrieval"])
            try:
                live = store.search(snap["namespace"], snap["query"], snap["top_k"], snap["threshold"])
            finally:
                store.close()
            assert [r["id"] for r in live] == [r["id"] for r in snap["records"]]
