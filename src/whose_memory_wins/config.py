"""Configuration loading. The resolved config is copied into every run manifest."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .env import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None, mode: str | None = None) -> dict[str, Any]:
    """Load the YAML config; ``mode`` forces mock providers or checks that real ones are set."""
    cfg = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if mode == "mock":
        cfg["decision_model"]["provider"] = "mock"
        cfg["ingestion_llm"]["provider"] = "mock"
    elif mode == "real":
        if cfg["decision_model"]["provider"] == "mock":
            raise ValueError("--mode real requested but the config's decision_model.provider is 'mock'")
    elif mode is not None:
        raise ValueError(f"mode must be 'mock' or 'real', got {mode!r}")
    return cfg


def mode_of(cfg: dict[str, Any]) -> str:
    """``mock`` if the decision model is the mock provider, else ``real``."""
    return "mock" if cfg["decision_model"]["provider"] == "mock" else "real"


def outputs_dir(cfg: dict[str, Any]) -> Path:
    """Absolute outputs directory from the config."""
    p = Path(cfg["paths"]["outputs"])
    return p if p.is_absolute() else PROJECT_ROOT / p


def credential_status() -> dict[str, Any]:
    """Report which credential sources exist WITHOUT revealing any secret."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return {
        "ANTHROPIC_API_KEY": "set" if key else "not set",
        "ANTHROPIC_AUTH_TOKEN": "set" if os.environ.get("ANTHROPIC_AUTH_TOKEN") else "not set",
        "dotenv_file": "present" if (PROJECT_ROOT / ".env").exists() else "absent",
    }


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """List of config problems (empty if valid). Real runs refuse to start with any."""
    errs = []
    dm = cfg["decision_model"]
    if dm["provider"] not in ("anthropic", "mock"):
        errs.append("decision_model.provider must be anthropic or mock")
    if dm["effort"] not in ("low", "medium", "high", "xhigh", "max"):
        errs.append("decision_model.effort invalid")
    if cfg["agent"]["max_model_calls"] < 1 or cfg["agent"]["max_tool_calls"] < 1:
        errs.append("agent budgets must be >= 1")
    if cfg["retries"]["max_attempts"] < 1:
        errs.append("retries.max_attempts must be >= 1")
    lim = cfg["limits"]
    for k in ("max_api_calls", "max_input_tokens", "max_output_tokens"):
        if not isinstance(lim.get(k), int) or lim[k] <= 0:
            errs.append(f"limits.{k} must be a positive integer (required ceiling)")
    pr = cfg["pricing"]
    if (pr.get("usd_per_million_input_tokens") is None) != (pr.get("usd_per_million_output_tokens") is None):
        errs.append("pricing: set both input and output prices, or neither")
    if pr.get("max_cost_usd") is not None:
        errs.append("pricing.max_cost_usd was renamed to pricing.stop_after_cost_usd (a stop-after threshold, "
                    "not a guaranteed cap)")
    thr = pr.get("stop_after_cost_usd", pr.get("max_cost_usd"))
    if thr is not None and (pr.get("usd_per_million_input_tokens") is None
                            or pr.get("usd_per_million_output_tokens") is None):
        errs.append("pricing.stop_after_cost_usd is set but prices are missing, so it cannot be enforced; "
                    "set both prices or remove the threshold")
    return errs


def redacted(cfg: dict[str, Any]) -> dict[str, Any]:
    """Config safe to store in manifests (it never contains secrets)."""
    return copy.deepcopy(cfg)  # config never contains secrets; keys come only from the environment
