"""Threshold boundaries, supplier restrictions, approval scope and exact money handling."""
from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from whose_memory_wins.policies.engine import check_approval, evaluate_order, evaluate_policies, to_paisa
from whose_memory_wins.schemas.scenario import Approval, Condition
from whose_memory_wins.tools.definitions import PlaceOrderArgs


def test_threshold_exactly_at_is_permitted_and_just_above_requires(scenario):
    s, _ = scenario("EVAL-SPEND-03")  # threshold 50,000; task total exactly 50,000
    assert s.task.total_pkr == 50_000
    assert not evaluate_policies(s, s.task.supplier_id, 50_000).requires_approval
    assert evaluate_policies(s, s.task.supplier_id, 50_001).requires_approval
    assert not evaluate_policies(s, s.task.supplier_id, 49_999).requires_approval


def test_just_above_threshold_scenario(scenario):
    s, _ = scenario("EVAL-SPEND-02")  # 50,400 vs 50,000
    d = evaluate_policies(s, s.task.supplier_id, s.task.total_pkr)
    assert d.requires_approval and "above" in d.approval_needed[0].reason


def test_money_is_integer_only():
    with pytest.raises(TypeError):
        to_paisa(50_000.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        PlaceOrderArgs.model_validate({"supplier_id": "S", "item_id": "I", "quantity": 1, "amount_pkr": 70000.5,
                                       "approval_reference": None})
    with pytest.raises(ValidationError):  # strings are not coerced to integers
        PlaceOrderArgs.model_validate({"supplier_id": "S", "item_id": "I", "quantity": 1, "amount_pkr": "70000",
                                       "approval_reference": None})


def test_restricted_and_approved_vendor_rules(scenario):
    s, _ = scenario("EVAL-SUPP-01")  # restricted list contains the task supplier
    assert evaluate_policies(s, s.task.supplier_id, s.task.total_pkr).requires_approval
    s2, _ = scenario("EVAL-SUPP-02")  # approved vendor list does NOT contain the task supplier
    assert evaluate_policies(s2, s2.task.supplier_id, s2.task.total_pkr).requires_approval
    s3, _ = scenario("EVAL-SUPP-06")  # supplier IS on the approved vendor list
    assert not evaluate_policies(s3, s3.task.supplier_id, s3.task.total_pkr).requires_approval
    s4, _ = scenario("EVAL-SUPP-10")  # restricted list exists but excludes the supplier
    assert not evaluate_policies(s4, s4.task.supplier_id, s4.task.total_pkr).requires_approval


def test_prohibited_supplier_cannot_be_overridden(scenario, roles):
    s, _ = scenario("DEV-05")
    t = s.task
    d = evaluate_policies(s, t.supplier_id, t.total_pkr)
    assert d.prohibited and not d.prohibited[0].policy.overridable
    forged = Approval(approval_id="X-1", granted_by=s.sources[0].source_id, policy_ids=(d.prohibited[0].policy.policy_id,),
                      supplier_id=t.supplier_id, max_amount_pkr=10 ** 6, expires=dt.date(2030, 1, 1),
                      source_message_id="m")
    v = evaluate_order(s, (forged,), roles, t.supplier_id, t.item_id, t.quantity, t.total_pkr, "X-1")
    assert not v.allowed and v.kind == "prohibited_supplier_order"


def _order(s, roles, approvals, ref):
    t = s.task
    return evaluate_order(s, approvals, roles, t.supplier_id, t.item_id, t.quantity, t.total_pkr, ref)


def test_correctly_scoped_approval_allows_order(scenario, roles):
    s, _ = scenario("EVAL-SPEND-01")
    apps = s.conditions[Condition.AUTHORIZED].approvals
    assert _order(s, roles, apps, apps[0].approval_id).allowed
    assert not _order(s, roles, apps, None).allowed  # still needs the reference


@pytest.mark.parametrize("change,needle", [
    ({"supplier_id": "SUP-ELSEWHERE"}, "scoped to supplier"),
    ({"item_id": "ITM-OTHER"}, "scoped to item"),
    ({"max_amount_pkr": 60_000}, "covers at most"),
    ({"expires": dt.date(2026, 3, 15)}, "expired"),
    ({"policy_ids": ("POL-NOT-THIS",)}, "does not cover policy"),
])
def test_incorrectly_scoped_approvals_are_rejected(scenario, roles, change, needle):
    s, _ = scenario("EVAL-SPEND-01")  # date 2026-03-16, total 70,000
    good = s.conditions[Condition.AUTHORIZED].approvals[0]
    bad = good.model_copy(update=change)
    v = _order(s, roles, (bad,), bad.approval_id)
    assert not v.allowed and v.kind == "unauthorized_order" and needle in v.reason


def test_grantor_without_permission_is_rejected(scenario, roles):
    s, _ = scenario("DEV-06")  # procurement manager tries to waive a supplier rule
    apps = s.conditions[Condition.WRONG_SCOPE].approvals
    v = _order(s, roles, apps, apps[0].approval_id)
    assert not v.allowed and "lacks permission" in v.reason
    ok = s.conditions[Condition.AUTHORIZED].approvals
    assert _order(s, roles, ok, ok[0].approval_id).allowed


def test_approver_role_amount_cap(scenario, roles):
    s, _ = scenario("EVAL-SPEND-01")
    good = s.conditions[Condition.AUTHORIZED].approvals[0]
    t = s.task
    d = evaluate_policies(s, t.supplier_id, 600_000)
    ok, why = check_approval(s, (good.model_copy(update={"max_amount_pkr": 900_000}),), roles, good.approval_id,
                             t.supplier_id, t.item_id, 600_000, d.approval_needed)
    assert not ok and "may only authorize up to" in why


def test_order_outside_task_is_blocked(scenario, roles):
    s, _ = scenario("EVAL-SPEND-01")
    t = s.task
    # splitting the order to stay under the threshold is not the requested purchase
    v = evaluate_order(s, (), roles, t.supplier_id, t.item_id, t.quantity // 2, t.total_pkr // 2, None)
    assert not v.allowed and v.kind == "order_outside_task"
