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
    mode: str = "real"
    name: str = "base"
    model: str = ""

    def create(self, system: str, messages: list[dict[str, Any]], tools: list[dict]) -> ProviderResponse:
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        return {}
