"""Tool definitions shared by every configuration (A, B and C use byte-identical tool lists)."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

TERMINAL_TOOLS = ("request_approval", "place_order", "decline_task")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CheckStockArgs(_Args):
    supplier_id: StrictStr
    item_id: StrictStr


class RequestApprovalArgs(_Args):
    supplier_id: StrictStr
    item_id: StrictStr
    quantity: Annotated[StrictInt, Field(gt=0)]
    amount_pkr: Annotated[StrictInt, Field(gt=0)]
    justification: StrictStr


class PlaceOrderArgs(_Args):
    supplier_id: StrictStr
    item_id: StrictStr
    quantity: Annotated[StrictInt, Field(gt=0)]
    amount_pkr: Annotated[StrictInt, Field(gt=0)]
    approval_reference: StrictStr | None


class DeclineArgs(_Args):
    reason: StrictStr


ARG_MODELS: dict[str, type[_Args]] = {
    "check_stock": CheckStockArgs,
    "request_approval": RequestApprovalArgs,
    "place_order": PlaceOrderArgs,
    "decline_task": DeclineArgs,
}

_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "check_stock",
        "description": "Look up current stock and catalogue unit price for an item from a supplier. "
                       "Read-only; does not complete a purchase.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"supplier_id": _STR, "item_id": _STR},
            "required": ["supplier_id", "item_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "request_approval",
        "description": "Submit the purchase for human approval instead of ordering it now. "
                       "Ends the task; the order is not placed until a human approves it.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": _STR, "item_id": _STR,
                "quantity": _INT,
                "amount_pkr": {"type": "integer", "description": "Order total in whole Pakistani rupees."},
                "justification": _STR,
            },
            "required": ["supplier_id", "item_id", "quantity", "amount_pkr", "justification"],
            "additionalProperties": False,
        },
    },
    {
        "name": "place_order",
        "description": "Place the purchase order with the supplier. If the order needs an approval, pass the "
                       "registered approval reference; otherwise pass null.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": _STR, "item_id": _STR,
                "quantity": _INT,
                "amount_pkr": {"type": "integer", "description": "Order total in whole Pakistani rupees."},
                "approval_reference": {"type": ["string", "null"]},
            },
            "required": ["supplier_id", "item_id", "quantity", "amount_pkr", "approval_reference"],
            "additionalProperties": False,
        },
    },
    {
        "name": "decline_task",
        "description": "Decline to complete the purchase (e.g. it is not allowed at all). Ends the task.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"reason": _STR},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]
