"""Provider interface for the decision model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TransientProviderError(Exception):
    """Rate limits, overload, 5xx, timeouts, connection errors: safe to retry the same request."""


class PermanentProviderError(Exception):
    """Bad request, auth, not found, etc.: retrying the same request will not help."""


@dataclass
class ProviderResponse:
    """Provider-neutral view of one model response (content, tool calls, tokens)."""
    stop_reason: str | None
    blocks: list[dict[str, Any]]            # JSON-serializable observable content blocks (for logging)
    assistant_content: Any                  # what to append as the assistant turn (passed back unchanged)
    tool_uses: list[dict[str, Any]]         # [{"id", "name", "input"}]
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class DecisionProvider:
    """Interface for the decision model. ``create`` sends one request; no retries inside."""
    mode: str = "real"
    name: str = "base"
    model: str = ""

    def create(self, system: str, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderResponse:
        """Send one request and return the response. Raise Transient/PermanentProviderError on failure."""
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        """Settings recorded in the run manifest (model, effort, SDK/API versions, ...)."""
        return {}
