"""Trial-level records: tool calls, model turns, scores and the full trial result."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["mock", "real"]
Config = Literal["A", "B", "C"]


class Loose(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolAction(Loose):
    """One tool call the model attempted, plus what the simulator did with it."""
    index: int
    model_call_index: int
    tool_use_id: str
    tool: str
    raw_input: dict[str, Any]
    valid_schema: bool
    validation_error: str | None = None
    executed: bool = False                 # True only if the simulator carried the action out
    blocked_reason: str | None = None      # why the simulator refused it (attempt still recorded)
    verdict_kind: str | None = None        # policy-engine classification (e.g. unauthorized_order)
    tool_result: str = ""


class ModelCall(Loose):
    index: int
    attempts: list[dict[str, Any]] = Field(default_factory=list)  # every attempt incl. transient failures
    stop_reason: str | None = None
    content: list[dict[str, Any]] = Field(default_factory=list)   # observable blocks (text, tool_use, thinking placeholder)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0


class Score(Loose):
    status: Literal["completed", "api_error", "step_limit", "invalid_output", "model_refusal", "budget_stop"]
    terminal_action: str | None = None     # request_approval | place_order | decline_task | None
    success: bool = False
    prohibited_attempt: bool = False
    prohibited_kinds: list[str] = Field(default_factory=list)
    unnecessary_approval: bool = False
    refusal: bool = False                  # decline_task when a legitimate action was available
    valid_output: bool = False             # trial produced a well-formed terminal action and no schema errors
    rule_applied: str = ""
    explanation: str = ""


class TrialResult(Loose):
    trial_id: str
    run_id: str
    mode: Mode
    experiment_kind: Literal["core", "ingestion", "diagnostic_removal"]
    scenario_id: str
    family: str
    condition: str
    config: Config
    repetition: int
    attempt: int = 1
    model: str
    started_at: str
    finished_at: str | None = None
    duration_s: float = 0.0
    context: dict[str, Any] = Field(default_factory=dict)       # exact model-visible system/messages/tools
    retrieval: dict[str, Any] = Field(default_factory=dict)     # retrieval snapshot reference + records
    policy_visible: bool = False
    model_calls: list[ModelCall] = Field(default_factory=list)
    actions: list[ToolAction] = Field(default_factory=list)
    final_text: str = ""
    score: Score | None = None
    labels: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
