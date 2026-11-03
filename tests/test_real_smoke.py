"""Tiny real-API smoke test (one decision trial). Costs a few cents at most.

Skipped (not passed) unless BOTH an Anthropic credential and WMW_RUN_REAL_TESTS=1 are present,
so that running the offline suite can never spend money by accident.
"""
from __future__ import annotations

import os

import pytest

from whose_memory_wins.config import load_config
from whose_memory_wins.experiments.runner import create_run, execute_run

pytestmark = [pytest.mark.real, pytest.mark.embeddings]

HAVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
OPTED_IN = os.environ.get("WMW_RUN_REAL_TESTS") == "1"


@pytest.mark.skipif(not (HAVE_KEY and OPTED_IN),
                    reason="real API smoke test needs ANTHROPIC_API_KEY and WMW_RUN_REAL_TESTS=1")
def test_one_real_trial(outputs, dev_fixture):
    cfg = load_config(mode="real")
    cfg["paths"]["outputs"] = str(outputs)
    cfg["limits"]["max_api_calls"] = 12
    run = create_run(cfg, name="smoke-real", split="development", experiment_kind="core",
                     snapshot_dir=dev_fixture / "retrieval", register_path=dev_fixture / "source_register.sqlite",
                     scenario_ids=["DEV-01"], conditions=["untrusted_claim"], configs=["C"], reps=1)
    st = execute_run(run, log=print)
    assert st["mode"] == "real"
    assert st["finished"] + st["failed_api_error"] == 1
    assert st["finished"] == 1, f"real call did not finish: {st}"
