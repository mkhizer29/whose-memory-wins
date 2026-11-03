from __future__ import annotations

import copy
from pathlib import Path

import pytest

from whose_memory_wins import env  # noqa: F401
from whose_memory_wins.config import load_config
from whose_memory_wins.dataset import load_roles, load_scenario
from whose_memory_wins.memory.fixtures import build_controlled_fixture


@pytest.fixture(scope="session")
def outputs(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("wmw_outputs")


@pytest.fixture(scope="session")
def mock_cfg(outputs) -> dict:
    cfg = load_config(mode="mock")
    cfg["paths"]["outputs"] = str(outputs)
    cfg["retries"]["backoff_s"] = [0, 0]
    return cfg


@pytest.fixture()
def cfg(mock_cfg) -> dict:
    return copy.deepcopy(mock_cfg)


@pytest.fixture(scope="session")
def roles():
    return load_roles()


@pytest.fixture(scope="session")
def dev_fixture(outputs, mock_cfg) -> Path:
    return build_controlled_fixture("development", outputs, mock_cfg["retrieval"], log=lambda s: None)


@pytest.fixture(scope="session")
def eval_fixture(outputs, mock_cfg) -> Path:
    return build_controlled_fixture("evaluation", outputs, mock_cfg["retrieval"], log=lambda s: None)


@pytest.fixture()
def scenario():
    def _get(sid):
        return load_scenario(sid)
    return _get
