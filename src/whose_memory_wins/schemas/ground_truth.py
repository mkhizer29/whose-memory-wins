"""Evaluator-only ground truth. Never imported by prompt construction code.

Stored in ``datasets/<split>/ground_truth/*.json``, separately from scenario files.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .scenario import Condition


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptableAction(Strict):
    """One acceptable terminal action and the arguments it must carry.

    ``approval_reference``: None = must be absent or is ignored (see ``reference_rule``).
    """
    tool: Literal["request_approval", "place_order", "decline_task"]
    supplier_id: str | None = None
    item_id: str | None = None
    quantity: int | None = None
    amount_pkr: int | None = None
    reference_rule: Literal["ignored", "must_equal"] = "ignored"
    approval_reference: str | None = None


class ConditionTruth(Strict):
    requires_approval_absent_exception: bool
    valid_exception_present: bool
    acceptable_terminal_actions: tuple[AcceptableAction, ...]
    # Kinds of prohibited actions for this condition. The scorer flags any attempted action
    # whose policy-engine verdict falls in one of these kinds.
    prohibited_kinds: tuple[Literal["unauthorized_order", "order_outside_task", "prohibited_supplier_order"], ...]
    unnecessary_approval_is_failure: bool
    explanation: str


class ScenarioTruth(Strict):
    scenario_id: str
    conflict_relevant: bool   # False for permitted-baseline controls where the claim cannot change the answer
    conditions: dict[Condition, ConditionTruth]
