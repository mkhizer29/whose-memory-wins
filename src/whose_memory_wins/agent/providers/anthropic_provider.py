"""Real decision model via the official Anthropic Python SDK (tested against anthropic==2.4.0).

Notes recorded in every run manifest:
* anthropic 2.4.0 sends API version 2026-10-21, which rejects temperature/top_p/top_k. Sampling
  therefore uses provider defaults and cannot be pinned.
* Claude Opus 5.5 always uses adaptive thinking; depth is controlled with output_config.effort.
  Thinking text is not requested (default display "omitted") and is never used for scoring.
* Forced tool_choice is not supported on this model, so tool_choice is "auto" with
  disable_parallel_tool_use=True (at most one tool call per model turn) and strict tool schemas.
* The SDK's own automatic retries are disabled (max_retries=0) so that every attempt is
  logged by the harness's bounded retry logic.
* Server-side refusal fallbacks are deliberately NOT enabled: they would silently switch the
  decision model mid-experiment. A refusal stop is recorded as a trial outcome instead.
"""
from __future__ import annotations

from typing import Any

from .base import DecisionProvider, PermanentProviderError, ProviderResponse, TransientProviderError


class AnthropicProvider(DecisionProvider):
    mode = "real"
    name = "anthropic"

    def __init__(self, model: str, effort: str, max_tokens: int, timeout_s: float):
        import anthropic

        self._anthropic = anthropic
        self._client = None  # created on first call, so manifests can be written without credentials
        self.timeout_s = timeout_s
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.sdk_version = anthropic.__version__

    def settings(self) -> dict[str, Any]:
        from anthropic._constants import ANTHROPIC_API_VERSION

        return {"provider": "anthropic", "model": self.model, "effort": self.effort, "max_tokens": self.max_tokens,
                "thinking": "adaptive (model default; not configurable off on this model)",
                "temperature": "not sent (rejected by API version; provider default sampling)",
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "sdk_version": self.sdk_version, "api_version": ANTHROPIC_API_VERSION}

    @property
    def client(self):
        if self._client is None:
            self._client = self._anthropic.Anthropic(max_retries=0, timeout=self.timeout_s)
        return self._client

    def create(self, system: str, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderResponse:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, tools=tools,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                output_config={"effort": self.effort},
            )
        except (a.RateLimitError, a.InternalServerError, a.APIConnectionError) as exc:  # APITimeoutError included
            raise TransientProviderError(f"{type(exc).__name__}: {exc}") from exc
        except a.APIStatusError as exc:
            if exc.status_code in (408, 409, 429) or exc.status_code >= 500:
                raise TransientProviderError(f"{type(exc).__name__} {exc.status_code}: {exc}") from exc
            raise PermanentProviderError(f"{type(exc).__name__} {exc.status_code}: {exc}") from exc
        except a.AnthropicError as exc:  # e.g. missing credentials at client construction
            raise PermanentProviderError(f"{type(exc).__name__}: {exc}") from exc

        blocks, tool_uses, texts = [], [], []
        for b in resp.content:
            d = b.model_dump(mode="json")
            if b.type == "thinking":
                # Keep only the fact that a thinking block existed; never use its content.
                blocks.append({"type": "thinking", "note": "thinking block present (content not used)"})
                continue
            if b.type == "redacted_thinking":
                blocks.append({"type": "redacted_thinking"})
                continue
            blocks.append(d)
            if b.type == "tool_use":
                tool_uses.append({"id": b.id, "name": b.name, "input": b.input})
            elif b.type == "text":
                texts.append(b.text)
        return ProviderResponse(
            stop_reason=resp.stop_reason, blocks=blocks, assistant_content=resp.content, tool_uses=tool_uses,
            text="\n".join(texts), input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens,
            model=resp.model, extra={"response_id": resp.id})
