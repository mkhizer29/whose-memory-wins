"""Bounded agent loop: model -> tool -> simulator -> model ..., with logged, bounded retries.

Retries are only for transient provider failures of a single request (same messages re-sent).
A valid model answer is never retried, whatever it says.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..schemas.records import ModelCall
from ..tools.definitions import TERMINAL_TOOLS
from ..tools.simulator import PurchasingSimulator
from .providers.base import DecisionProvider, PermanentProviderError, TransientProviderError


class BudgetExceeded(Exception):
    pass


INPUT_OVERHEAD_TOKENS = 2000  # tool-use framing and other provider-side additions to the request


def estimate_input_tokens(system: str, messages: list[dict[str, Any]], tools: list[dict]) -> int:
    """Conservative upper-bound estimate of a request's input tokens: UTF-8 bytes of the serialized
    request (a tokenizer yields at most about one token per byte) plus a fixed overhead."""
    def enc(o):
        return o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o)
    payload = json.dumps({"system": system, "messages": messages, "tools": tools}, default=enc)
    return len(payload.encode("utf-8")) + INPUT_OVERHEAD_TOKENS


@dataclass
class LoopResult:
    # completed | no_decision | api_error | step_limit | invalid_output | model_refusal | budget_stop
    status: str
    model_calls: list[ModelCall] = field(default_factory=list)
    final_text: str = ""
    error: str | None = None


def run_agent(provider: DecisionProvider, context: dict[str, Any], sim: PurchasingSimulator, *,
              max_model_calls: int, max_tool_calls: int, max_attempts: int, backoff_s: list[float],
              before_call: Callable[[int, int], None] = lambda est_in, max_out: None,
              after_call: Callable[..., None] = lambda i, o, failed=False: None,
              sleep: Callable[[float], None] = time.sleep) -> LoopResult:
    messages: list[dict[str, Any]] = [dict(m) for m in context["messages"]]
    result = LoopResult(status="completed")
    tool_calls = 0
    max_out = int(getattr(provider, "max_tokens", 0) or 0)  # the API never returns more output than this
    for call_index in range(max_model_calls):
        mc = ModelCall(index=call_index)
        result.model_calls.append(mc)
        resp = None
        for attempt in range(1, max_attempts + 1):
            try:
                before_call(estimate_input_tokens(context["system"], messages, context["tools"]), max_out)
            except BudgetExceeded as exc:
                mc.attempts.append({"attempt": attempt, "ok": False, "error_type": "BudgetExceeded",
                                    "message": str(exc), "at": _now()})
                result.status, result.error = "budget_stop", str(exc)
                return result
            t0 = time.perf_counter()
            try:
                resp = provider.create(context["system"], messages, context["tools"])
                mc.attempts.append({"attempt": attempt, "ok": True, "at": _now()})
                mc.duration_s = round(time.perf_counter() - t0, 3)
                break
            except TransientProviderError as exc:
                mc.attempts.append({"attempt": attempt, "ok": False, "error_type": "transient",
                                    "message": str(exc)[:500], "at": _now()})
                after_call(0, 0, failed=True)  # counted: a failed request may still be billed
                if attempt < max_attempts:
                    sleep(backoff_s[min(attempt - 1, len(backoff_s) - 1)])
            except PermanentProviderError as exc:
                mc.attempts.append({"attempt": attempt, "ok": False, "error_type": "permanent",
                                    "message": str(exc)[:500], "at": _now()})
                after_call(0, 0, failed=True)
                result.status, result.error = "api_error", str(exc)[:500]
                return result
        if resp is None:
            result.status = "api_error"
            result.error = f"transient failures on all {max_attempts} attempts"
            return result

        mc.stop_reason, mc.content = resp.stop_reason, resp.blocks
        mc.input_tokens, mc.output_tokens = resp.input_tokens, resp.output_tokens
        after_call(resp.input_tokens, resp.output_tokens)
        if resp.text:
            result.final_text = resp.text
        if resp.stop_reason == "refusal":
            result.status = "model_refusal"
            return result
        if not resp.tool_uses:
            # The model chose to stop. "completed" requires that it attempted at least one
            # schema-valid purchasing decision (executed or blocked); see evaluation.scorer.
            invalid = any(not a.valid_schema for a in sim.actions)
            decided = any(a.valid_schema and a.tool in TERMINAL_TOOLS for a in sim.actions)
            if resp.stop_reason == "max_tokens" or (invalid and not decided):
                result.status = "invalid_output"
            elif not decided:
                result.status = "no_decision"   # e.g. a text-only reply, or only a stock lookup
            return result

        messages.append({"role": "assistant", "content": resp.assistant_content})
        tool_results = []
        for tu in resp.tool_uses:
            tool_calls += 1
            action = sim.handle(tu["name"], tu["input"], tu["id"], call_index)
            tool_results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": action.tool_result,
                                 **({"is_error": True} if not action.executed else {})})
        if sim.terminal is not None:
            return result
        if tool_calls >= max_tool_calls:
            result.status = "step_limit"
            return result
        messages.append({"role": "user", "content": tool_results})
    result.status = "step_limit"
    return result


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
