from .base import DecisionProvider, PermanentProviderError, ProviderResponse, TransientProviderError


def make_provider(cfg: dict) -> DecisionProvider:
    """Build the decision model from the ``decision_model`` config section. Never falls back silently."""
    if cfg["provider"] == "mock":
        from .mock_provider import MockProvider
        return MockProvider(cfg.get("mock_policy", "gullible"))
    if cfg["provider"] == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(cfg["model"], cfg["effort"], cfg["max_tokens"], cfg["request_timeout_s"])
    raise ValueError(f"unknown provider {cfg['provider']!r}")


__all__ = ["DecisionProvider", "PermanentProviderError", "ProviderResponse", "TransientProviderError", "make_provider"]
