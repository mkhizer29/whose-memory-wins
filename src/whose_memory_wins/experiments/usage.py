"""Cumulative API usage per run with hard ceilings (calls, tokens, optional USD)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent.loop import BudgetExceeded


class UsageTracker:
    def __init__(self, path: Path, limits: dict[str, Any], pricing: dict[str, Any]):
        self.path = Path(path)
        self.limits = limits
        self.pricing = pricing
        self.state = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0}
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self) -> None:
        self.path.write_text(json.dumps({**self.state, "cost_usd": self.cost_usd()}, indent=2), encoding="utf-8")

    def cost_usd(self, input_tokens: int | None = None, output_tokens: int | None = None) -> float | None:
        pi, po = self.pricing.get("usd_per_million_input_tokens"), self.pricing.get("usd_per_million_output_tokens")
        if pi is None or po is None:
            return None
        i = self.state["input_tokens"] if input_tokens is None else input_tokens
        o = self.state["output_tokens"] if output_tokens is None else output_tokens
        return round(i / 1e6 * pi + o / 1e6 * po, 6)

    def exceeded(self) -> str | None:
        s, lim = self.state, self.limits
        if s["api_calls"] >= lim["max_api_calls"]:
            return f"API call ceiling reached ({s['api_calls']}/{lim['max_api_calls']})"
        if s["input_tokens"] >= lim["max_input_tokens"]:
            return f"input token ceiling reached ({s['input_tokens']}/{lim['max_input_tokens']})"
        if s["output_tokens"] >= lim["max_output_tokens"]:
            return f"output token ceiling reached ({s['output_tokens']}/{lim['max_output_tokens']})"
        cap, cost = self.pricing.get("max_cost_usd"), self.cost_usd()
        if cap is not None and cost is not None and cost >= cap:
            return f"cost ceiling reached (USD {cost:.4f}/{cap})"
        return None

    def can_start_trial(self, worst_case_calls: int) -> str | None:
        reason = self.exceeded()
        if reason:
            return reason
        if self.state["api_calls"] + worst_case_calls > self.limits["max_api_calls"]:
            return (f"not enough call budget left for a worst-case trial ({worst_case_calls} calls; "
                    f"{self.limits['max_api_calls'] - self.state['api_calls']} remaining)")
        return None

    def before_call(self) -> None:
        reason = self.exceeded()
        if reason:
            raise BudgetExceeded(reason)

    def after_call(self, input_tokens: int, output_tokens: int) -> None:
        self.state["api_calls"] += 1
        self.state["input_tokens"] += input_tokens
        self.state["output_tokens"] += output_tokens
        self._save()
