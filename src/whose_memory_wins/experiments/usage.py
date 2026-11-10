"""Cumulative decision-model API usage per run, with ceilings checked BEFORE each request.

What is enforced, and how strictly (also shown by ``describe()`` in the CLI and in usage.json):

* ``max_api_calls`` - hard. Every request attempt counts, including transient failures that are
  retried (a failed request may still have been billed), so retries cannot exceed the ceiling.
* ``max_output_tokens`` - hard. Before each request the tracker reserves the request's
  ``max_tokens`` (the API never returns more output than that), and does not send the request
  if used + reserved would exceed the ceiling.
* ``max_input_tokens`` - conservative estimate. Before each request the tracker reserves an
  upper-bound ESTIMATE of the input size (UTF-8 bytes of the serialized request plus a fixed
  overhead; tokenizers produce at most about one token per byte). This is not a guarantee.
* ``stop_after_cost_usd`` - a STOP-AFTER THRESHOLD, not a guaranteed cap. It is only active when
  both prices are configured. The same reservations are converted to USD, so a request is not
  started if its estimated worst case would cross the threshold; but the prices are entered by
  hand and the input size is estimated, so actual billing can differ. Without prices, no dollar
  limit is enforced and the harness says so.

Mem0 ingestion (Track 2) is NOT counted here: see ``memory.mock_llm.CountingLLM``. It has a hard
call ceiling only; Mem0 does not report token usage, so no token or dollar limit applies to it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agent.loop import BudgetExceeded


def dollar_threshold(pricing: dict[str, Any]) -> float | None:
    """The configured stop-after dollar threshold, or None."""
    # ``max_cost_usd`` is the pre-fix name of the same setting (kept readable for old manifests).
    v = pricing.get("stop_after_cost_usd")
    return v if v is not None else pricing.get("max_cost_usd")


def prices_known(pricing: dict[str, Any]) -> bool:
    """True only if both input and output prices are configured."""
    return (pricing.get("usd_per_million_input_tokens") is not None
            and pricing.get("usd_per_million_output_tokens") is not None)


def describe_limits(limits: dict[str, Any], pricing: dict[str, Any]) -> dict[str, str]:
    """Plain-language description of which limits are enforced and how strictly."""
    thr = dollar_threshold(pricing)
    if thr is None:
        dollar = "none configured"
    elif not prices_known(pricing):
        dollar = f"USD {thr} configured but NOT ENFORCED: prices are missing"
    else:
        dollar = (f"stop-after threshold USD {thr} (not a guaranteed cap: prices are hand-entered and input "
                  "size is estimated)")
    return {
        "api_calls": f"hard ceiling {limits['max_api_calls']} (retried attempts count)",
        "output_tokens": f"hard ceiling {limits['max_output_tokens']} (each request reserves its max_tokens first)",
        "input_tokens": f"ceiling {limits['max_input_tokens']} using a conservative pre-request estimate",
        "cost_usd": dollar,
        "scope": "decision-model calls of this run only; Mem0 ingestion calls are counted separately",
    }


class UsageTracker:
    """Cumulative decision-model usage of one run, persisted to usage.json after every call."""
    def __init__(self, path: Path, limits: dict[str, Any], pricing: dict[str, Any]):
        self.path = Path(path)
        self.limits = limits
        self.pricing = pricing
        self.state = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "failed_requests": 0}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.state.update({k: saved[k] for k in self.state if k in saved})

    def _save(self) -> None:
        self.path.write_text(json.dumps({**self.state, "cost_usd": self.cost_usd(),
                                         "limits_enforced": describe_limits(self.limits, self.pricing),
                                         "failed_request_tokens": "unknown (counted as 0 tokens; the call "
                                                                  "itself is counted)"}, indent=2),
                             encoding="utf-8")

    def cost_usd(self, input_tokens: int | None = None, output_tokens: int | None = None) -> float | None:
        """Estimated cost in USD from configured prices, or None if prices are missing."""
        if not prices_known(self.pricing):
            return None
        pi, po = self.pricing["usd_per_million_input_tokens"], self.pricing["usd_per_million_output_tokens"]
        i = self.state["input_tokens"] if input_tokens is None else input_tokens
        o = self.state["output_tokens"] if output_tokens is None else output_tokens
        return round(i / 1e6 * pi + o / 1e6 * po, 6)

    def exceeded(self) -> str | None:
        """Reason a ceiling has already been reached, or None."""
        s, lim = self.state, self.limits
        if s["api_calls"] >= lim["max_api_calls"]:
            return f"API call ceiling reached ({s['api_calls']}/{lim['max_api_calls']})"
        if s["input_tokens"] >= lim["max_input_tokens"]:
            return f"input token ceiling reached ({s['input_tokens']}/{lim['max_input_tokens']})"
        if s["output_tokens"] >= lim["max_output_tokens"]:
            return f"output token ceiling reached ({s['output_tokens']}/{lim['max_output_tokens']})"
        thr, cost = dollar_threshold(self.pricing), self.cost_usd()
        if thr is not None and cost is not None and cost >= thr:
            return f"stop-after cost threshold reached (USD {cost:.4f}/{thr})"
        return None

    def reservation_problem(self, est_input_tokens: int, max_output_tokens: int) -> str | None:
        """Why the next request must not be sent, given its worst-case size; None if it may be sent."""
        s, lim = self.state, self.limits
        if s["api_calls"] + 1 > lim["max_api_calls"]:
            return f"API call ceiling would be exceeded ({s['api_calls']} + 1 > {lim['max_api_calls']})"
        if s["input_tokens"] + est_input_tokens > lim["max_input_tokens"]:
            return (f"input token ceiling would be exceeded by the next request (used {s['input_tokens']} + "
                    f"estimated {est_input_tokens} > {lim['max_input_tokens']})")
        if s["output_tokens"] + max_output_tokens > lim["max_output_tokens"]:
            return (f"output token ceiling would be exceeded by the next request (used {s['output_tokens']} + "
                    f"reserved max_tokens {max_output_tokens} > {lim['max_output_tokens']})")
        thr = dollar_threshold(self.pricing)
        worst = self.cost_usd(s["input_tokens"] + est_input_tokens, s["output_tokens"] + max_output_tokens)
        if thr is not None and worst is not None and worst > thr:
            return f"stop-after cost threshold would be crossed by the next request (worst case USD {worst:.4f} > {thr})"
        return None

    def can_start_trial(self, worst_case_calls: int) -> str | None:
        """Reason a new trial must not start (ceilings, or too little call budget left), or None."""
        reason = self.exceeded()
        if reason:
            return reason
        if self.state["api_calls"] + worst_case_calls > self.limits["max_api_calls"]:
            return (f"not enough call budget left for a worst-case trial ({worst_case_calls} calls; "
                    f"{self.limits['max_api_calls'] - self.state['api_calls']} remaining)")
        return None

    def before_call(self, est_input_tokens: int = 0, max_output_tokens: int = 0) -> None:
        """Raise ``BudgetExceeded`` unless the next request, at its worst-case size, fits every limit."""
        reason = self.exceeded() or self.reservation_problem(est_input_tokens, max_output_tokens)
        if reason:
            raise BudgetExceeded(reason)

    def after_call(self, input_tokens: int, output_tokens: int, failed: bool = False) -> None:
        """Count one request attempt and its reported tokens (failed attempts count as calls)."""
        self.state["api_calls"] += 1
        self.state["input_tokens"] += input_tokens
        self.state["output_tokens"] += output_tokens
        if failed:
            self.state["failed_requests"] += 1
        self._save()
