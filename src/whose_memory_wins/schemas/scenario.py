"""Scenario data: the application-side facts of one purchasing situation.

Nothing in this module is ground truth. Expected outcomes live in ``ground_truth.py``
and are stored in separate files so that prompt construction never touches them.
Money is always an integer number of Pakistani rupees (whole PKR) in the dataset and
integer paisa (1/100 PKR) inside the policy engine; no floats are used anywhere.
"""
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Family(str, Enum):
    SPENDING = "spending_threshold"
    SUPPLIER = "supplier_restriction"


class Condition(str, Enum):
    CLEAN = "clean"
    UNTRUSTED = "untrusted_claim"
    AUTHORIZED = "authorized_exception"
    # Development-only conditions used to exercise scope handling.
    IMPERSONATION = "impersonation_claim"
    WRONG_SCOPE = "wrong_scope_exception"
    EXPIRED = "expired_exception"


CORE_CONDITIONS = (Condition.CLEAN, Condition.UNTRUSTED, Condition.AUTHORIZED)
CONFLICTING_CONDITIONS = (Condition.UNTRUSTED, Condition.IMPERSONATION, Condition.WRONG_SCOPE, Condition.EXPIRED)


class Company(Strict):
    name: str
    current_date: dt.date


class Task(Strict):
    """The purchasing request. ``text`` is shown to the model verbatim in every configuration."""
    text: str
    supplier_id: str
    item_id: str
    quantity: int = Field(gt=0)
    unit_price_pkr: int = Field(gt=0)
    total_pkr: int = Field(gt=0)
    retrieval_query: str  # the query sent to memory search (fixed across A/B/C)


class SpendingRule(Strict):
    type: Literal["spending_threshold"] = "spending_threshold"
    threshold_pkr: int = Field(gt=0)  # approval required when total is strictly ABOVE this value


class RestrictedSupplierRule(Strict):
    type: Literal["restricted_suppliers"] = "restricted_suppliers"
    supplier_ids: tuple[str, ...]      # orders from these suppliers require approval


class ApprovedVendorRule(Strict):
    type: Literal["approved_vendor_list"] = "approved_vendor_list"
    supplier_ids: tuple[str, ...]      # orders from suppliers NOT on this list require approval


class ProhibitedSupplierRule(Strict):
    type: Literal["prohibited_suppliers"] = "prohibited_suppliers"
    supplier_ids: tuple[str, ...]      # never order from these (non-overridable)


Rule = SpendingRule | RestrictedSupplierRule | ApprovedVendorRule | ProhibitedSupplierRule


class Policy(Strict):
    policy_id: str
    family: Family
    title: str
    text: str                 # human-readable policy text (stored in memory and in the policy register)
    rule: Rule = Field(discriminator="type")
    overridable: bool
    scope: Literal["all_purchases"] = "all_purchases"


class Item(Strict):
    item_id: str
    name: str
    unit_price_pkr: int = Field(gt=0)
    stock: int = Field(ge=0)


class Supplier(Strict):
    supplier_id: str
    name: str
    items: tuple[Item, ...]


class Source(Strict):
    """A message sender. ``role`` is assigned by the application/test harness, never parsed from text."""
    source_id: str
    display_name: str
    role: str


class MemoryFact(Strict):
    """An ordinary record that goes into memory (policy text, catalogue line, distractor note)."""
    record_key: str
    source_id: str
    message_text: str   # the original input message (used by normal ingestion)
    memory_text: str    # the controlled memory record text (used by controlled insertion)
    kind: Literal["policy", "business", "distractor"]
    policy_id: str | None = None


class ClaimedAuthorization(Strict):
    """What a message CLAIMS. Claims never create approvals; only ``Approval`` records do."""
    reference: str
    supplier_id: str
    policy_ids: tuple[str, ...]
    max_amount_pkr: int
    expires: dt.date


class ConditionMessage(Strict):
    message_id: str
    source_id: str
    message_text: str
    memory_text: str
    claimed_authorization: ClaimedAuthorization | None = None


class Approval(Strict):
    """An authorization record in the application-owned register. Only these can authorize."""
    approval_id: str
    granted_by: str               # source_id
    policy_ids: tuple[str, ...]
    supplier_id: str
    item_id: str | None = None
    max_amount_pkr: int = Field(gt=0)
    expires: dt.date
    source_message_id: str


class ConditionSpec(Strict):
    messages: tuple[ConditionMessage, ...] = ()
    approvals: tuple[Approval, ...] = ()


class Scenario(Strict):
    scenario_id: str
    split: Literal["development", "evaluation"]
    family: Family
    design_note: str              # why the scenario exists (shown in the dashboard; not in prompts)
    company: Company
    task: Task
    policies: tuple[Policy, ...]
    suppliers: tuple[Supplier, ...]
    sources: tuple[Source, ...]
    facts: tuple[MemoryFact, ...]
    conditions: dict[Condition, ConditionSpec]

    def source(self, source_id: str) -> Source:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)

    def supplier(self, supplier_id: str) -> Supplier | None:
        return next((s for s in self.suppliers if s.supplier_id == supplier_id), None)

    def item(self, supplier_id: str, item_id: str) -> Item | None:
        sup = self.supplier(supplier_id)
        if sup is None:
            return None
        return next((i for i in sup.items if i.item_id == item_id), None)


class RoleDefinition(Strict):
    role: str
    description: str
    # e.g. "grant_exception:spending_threshold", "grant_exception:restricted_suppliers"
    permissions: tuple[str, ...]
    max_exception_amount_pkr: int | None = None   # None = no amount cap (only meaningful with permissions)


class RoleCatalog(Strict):
    roles: tuple[RoleDefinition, ...]

    def get(self, role: str) -> RoleDefinition:
        for r in self.roles:
            if r.role == role:
                return r
        raise KeyError(f"Unknown role {role!r}")
