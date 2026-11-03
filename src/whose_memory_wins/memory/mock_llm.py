"""MOCK extraction LLM for offline verification of the Track 2 pipeline. Never research data.

Swapped in for Mem0's LLM object (``Memory.llm``) only in mock mode, so the rest of Mem0's
normal ``add(infer=True)`` pipeline (context gathering, parsing, hashing, embedding, storage,
history) still runs. It returns each new user message as one memory, with common
sender-introduction prefixes removed (a crude imitation of attribution loss).
"""
from __future__ import annotations

import json
import re

_PREFIXES = re.compile(r"^(Note for the purchasing assistant:\s*|Policy update \([^)]*\):\s*|Catalogue note:\s*|"
                       r"For your records:\s*|Update on [^:]+:\s*|This is [^.]+\.\s*)+")


class MockExtractionLLM:
    is_mock = True

    def __init__(self):
        self.calls = 0

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
        self.calls += 1
        prompt = messages[-1]["content"]
        new = prompt.split("## New Messages", 1)[-1].split("## Observation Date", 1)[0]
        memories = []
        for line in new.strip().splitlines():
            if line.startswith("user: "):
                text = _PREFIXES.sub("", line[len("user: "):].strip())
                if text:
                    memories.append({"id": str(len(memories)), "text": text})
        return json.dumps({"memory": memories})


class CountingLLM:
    """Wraps Mem0's LLM to count calls and enforce the run's API-call ceiling."""

    def __init__(self, inner, max_calls: int):
        self.inner = inner
        self.max_calls = max_calls
        self.calls = 0

    def generate_response(self, *args, **kwargs):
        if self.calls >= self.max_calls:
            from ..agent.loop import BudgetExceeded
            raise BudgetExceeded(f"ingestion LLM call ceiling reached ({self.calls}/{self.max_calls})")
        self.calls += 1
        return self.inner.generate_response(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)
