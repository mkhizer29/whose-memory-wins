"""Protected, application-owned policy register (used by configurations B and C).

Selection uses only the task context (company and purchase scope). It never looks at the
amount relative to a threshold, the condition, approvals or ground truth, so it cannot
leak the expected answer: every policy whose scope covers the purchase is returned.
"""
from __future__ import annotations

from ..schemas.scenario import Policy, Scenario


def applicable_policies(scenario: Scenario) -> list[Policy]:
    return sorted((p for p in scenario.policies if p.scope == "all_purchases"), key=lambda p: p.policy_id)


def render_policy_register(scenario: Scenario) -> str:
    lines = []
    for p in applicable_policies(scenario):
        tag = "can be overridden only by a registered, authorized exception" if p.overridable else "cannot be overridden"
        lines.append(f"- [{p.policy_id}] {p.title}: {p.text} ({tag})")
    return "\n".join(lines)
