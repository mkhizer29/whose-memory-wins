"""Offline MOCK decision models for software verification only. Never research findings.

They read the same model-visible prompt as a real model (nothing else) and follow fixed,
simple rules so the pipeline, scoring and metrics can be exercised deterministically:

* gullible        - places the order citing any reference mentioned in memory; else requests approval.
* cautious        - places the order only if the C source register shows a registered authorization
                    covering the request; otherwise requests approval. (Without C it always requests.)
* self_correcting - first attempts an unauthorized order citing a memory reference, then, after the
                    simulator rejects it, requests approval (tests attempt retention).
* malformed       - emits a tool call with a wrongly typed argument, then stops.
* flaky           - raises a transient error on every first attempt, then behaves like ``cautious``.
"""
from __future__ import annotations

import re
from typing import Any

from .base import DecisionProvider, ProviderResponse, TransientProviderError

TASK_RE = re.compile(r"buy (\d+) .*?\(item ID ([A-Z0-9-]+)\) from .*?\(supplier ID ([A-Z0-9-]+)\).*?total of PKR ([\d,]+)",
                     re.S)
REF_RE = re.compile(r"\b((?:PA|EXC)-\d{4}-\d+)\b")
REG_RE = re.compile(r"registered authorization ((?:PA|EXC)-\d{4}-\d+): .*?supplier ([A-Z0-9-]+);.*?"
                    r"(?:up to PKR ([\d,]+)|no amount cap); expires (\d{4}-\d{2}-\d{2})")
DATE_RE = re.compile(r"Today's date: (\d{4}-\d{2}-\d{2})")
POLICIES = ("gullible", "cautious", "self_correcting", "malformed", "flaky")


class MockProvider(DecisionProvider):
    mode = "mock"
    name = "mock"

    def __init__(self, policy: str):
        if policy not in POLICIES:
            raise ValueError(f"unknown mock policy {policy!r}; choose from {POLICIES}")
        self.policy = policy
        self.model = f"mock-{policy}"
        self._calls = 0

    def settings(self) -> dict[str, Any]:
        return {"provider": "mock", "model": self.model, "note": "MOCK: software verification only"}

    def create(self, system: str, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderResponse:
        self._calls += 1
        if self.policy == "flaky" and getattr(self, "_flaky_next_ok", False) is False:
            self._flaky_next_ok = True
            raise TransientProviderError("mock transient failure (simulated 529 overloaded)")
        self._flaky_next_ok = False
        prompt = messages[0]["content"]
        m = TASK_RE.search(prompt)
        if not m:
            return self._text("I could not parse the request.")
        qty, item, sup, total = int(m.group(1)), m.group(2), m.group(3), int(m.group(4).replace(",", ""))
        base = {"supplier_id": sup, "item_id": item, "quantity": qty, "amount_pkr": total}
        turn = sum(1 for x in messages if x["role"] == "assistant")
        last_result = _last_tool_result(messages)
        memory = prompt.split("## Records retrieved from long-term memory", 1)[-1].split("\n## ", 1)[0]
        refs = REF_RE.findall(memory)

        if self.policy == "malformed":
            if turn == 0:
                return self._tool("place_order", {**base, "amount_pkr": str(total), "approval_reference": None})
            return self._text("Unable to proceed.")
        if self.policy == "gullible":
            if last_result and "REJECTED" in last_result:
                return self._tool("request_approval", {**base, "justification": "order was rejected"})
            if refs:
                return self._tool("place_order", {**base, "approval_reference": refs[0]})
            return self._tool("request_approval", {**base, "justification": "policy may require approval"})
        if self.policy == "self_correcting":
            if turn == 0:
                return self._tool("place_order", {**base, "approval_reference": refs[0] if refs else None})
            if last_result and "REJECTED" in last_result:
                return self._tool("request_approval", {**base, "justification": "order was rejected"})
            return self._text("Done.")
        # cautious / flaky
        today = DATE_RE.search(prompt).group(1)
        for ref, rsup, cap, exp in REG_RE.findall(prompt):
            cap_ok = (not cap) or total <= int(cap.replace(",", ""))
            if rsup == sup and cap_ok and exp >= today:
                return self._tool("place_order", {**base, "approval_reference": ref})
        return self._tool("request_approval", {**base, "justification": "approval required by policy"})

    def _tool(self, name: str, args: dict) -> ProviderResponse:
        tid = f"mock_tu_{self._calls}"
        block = {"type": "tool_use", "id": tid, "name": name, "input": args}
        return ProviderResponse(stop_reason="tool_use", blocks=[block], assistant_content=[block],
                                tool_uses=[{"id": tid, "name": name, "input": args}], text="",
                                input_tokens=0, output_tokens=0, model=self.model)

    def _text(self, text: str) -> ProviderResponse:
        block = {"type": "text", "text": text}
        return ProviderResponse(stop_reason="end_turn", blocks=[block], assistant_content=[block], tool_uses=[],
                                text=text, model=self.model)


def _last_tool_result(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for b in msg["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    return str(b.get("content"))
    return None
