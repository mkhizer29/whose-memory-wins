"""Dataset consistency, balance, unique IDs, and that the validator actually catches errors."""
from __future__ import annotations

import json
import shutil

from whose_memory_wins.dataset import DATASETS, load_roles, load_split, validate_all, validate_split
from whose_memory_wins.schemas.scenario import CORE_CONDITIONS, Family


def test_dataset_validates_cleanly():
    assert validate_all() == {"development": [], "evaluation": []}


def test_evaluation_balance_and_unique_ids():
    scenarios, truths = load_split("evaluation")
    assert len(scenarios) == 24
    assert sum(s.family == Family.SPENDING for s in scenarios) == 12
    assert sum(s.family == Family.SUPPLIER for s in scenarios) == 12
    assert len({s.scenario_id for s in scenarios}) == 24
    for s in scenarios:
        assert set(s.conditions) == set(CORE_CONDITIONS)
    kinds = {truths[s.scenario_id].conflict_relevant for s in scenarios}
    assert kinds == {True, False}  # clean includes approval-required AND permitted purchases
    dev, _ = load_split("development")
    assert len(dev) == 6


def _mutated(tmp_path, split, sid, mutate):
    root = tmp_path / "datasets"
    shutil.copytree(DATASETS, root)
    p = root / split / "ground_truth" / f"{sid}.json"
    d = json.loads(p.read_text())
    mutate(d)
    p.write_text(json.dumps(d))
    scenarios, truths = load_split(split, root)
    return validate_split(split, scenarios, truths, load_roles())


def test_validator_catches_wrong_expected_action(tmp_path):
    def flip(d):  # claim the untrusted case should place the order
        a = d["conditions"]["untrusted_claim"]["acceptable_terminal_actions"][0]
        a["tool"] = "place_order"
    errs = _mutated(tmp_path, "evaluation", "EVAL-SPEND-01", flip)
    assert any("not among the acceptable actions" in e for e in errs)


def test_validator_catches_inconsistent_flags(tmp_path):
    def flip(d):
        d["conditions"]["clean"]["requires_approval_absent_exception"] = False
    errs = _mutated(tmp_path, "evaluation", "EVAL-SUPP-03", flip)
    assert any("disagrees with the policy engine" in e for e in errs)


def test_validator_catches_unsolvable_authorized_condition(tmp_path):
    root = tmp_path / "datasets"
    shutil.copytree(DATASETS, root)
    p = root / "evaluation" / "scenarios" / "EVAL-SPEND-04.json"
    d = json.loads(p.read_text())
    d["conditions"]["authorized_exception"]["approvals"][0]["max_amount_pkr"] = 1000  # too small
    p.write_text(json.dumps(d))
    scenarios, truths = load_split("evaluation", root)
    errs = validate_split("evaluation", scenarios, truths, load_roles())
    assert any("not valid (unsolvable" in e for e in errs)


def test_untrusted_and_authorized_claims_are_matched():
    scenarios, _ = load_split("evaluation")
    for s in scenarios:
        u = s.conditions[CORE_CONDITIONS[1]].messages[0]
        a = s.conditions[CORE_CONDITIONS[2]].messages[0]
        assert u.message_text == a.message_text and u.memory_text == a.memory_text
        assert u.source_id != a.source_id
        assert not s.conditions[CORE_CONDITIONS[1]].approvals  # a claim never creates an approval
