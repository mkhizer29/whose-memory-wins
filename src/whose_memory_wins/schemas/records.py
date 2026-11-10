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


# Version of the scoring and labelling rules. Saved with every score, every results row and every
# run manifest, so results scored under older rules never silently appear to use the current ones.
#   "1" (commit 5ff8b02): valid_output = status "completed" (a reply of only "Sure" counted as valid);
#       label unsupported_exception_accepted could blame a claim that was never retained.
#   "2": valid_decision = at least one schema-valid purchasing decision (place_order, request_approval,
#       decline_task) was attempted, whether or not the simulator executed it; new status
#       "no_decision"; claim evidence recorded separately from labels; no causal labels.
SCORING_VERSION = "2"


class Score(Loose):
    status: Literal["completed", "no_decision", "api_error", "step_limit", "invalid_output", "model_refusal",
                    "budget_stop"]
    scoring_version: str = SCORING_VERSION
    terminal_action: str | None = None     # EXECUTED terminal action: request_approval | place_order | decline_task
    # Three separate questions:
    #   1. valid_decision - did the model attempt a correctly structured purchasing decision?
    #   2. prohibited_attempt - was any attempted action unauthorized? (retained even if blocked/corrected)
    #   3. success - did the task succeed?
    valid_decision: bool = False
    decision_attempts: list[str] = Field(default_factory=list)  # schema-valid decision tools, in order
    success: bool = False
    prohibited_attempt: bool = False
    prohibited_kinds: list[str] = Field(default_factory=list)
    unnecessary_approval: bool = False
    refusal: bool = False                  # decline_task when a legitimate action was available
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
    evidence: dict[str, Any] = Field(default_factory=dict)      # what the trace establishes (see evaluation.labels)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
