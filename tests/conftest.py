"""pytest 共享夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodrisk.config import Config  # noqa: E402
from floodrisk.io import load_dataset  # noqa: E402
from floodrisk.synthetic import write_synthetic_bundle  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def dyn_bundle(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("syn_dynamic")
    return write_synthetic_bundle(out, mode="dynamic", n_grids=24, n_steps=72, seed=7)


@pytest.fixture(scope="session")
def static_bundle(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("syn_static")
    return write_synthetic_bundle(out, mode="static", n_grids=36, seed=7)


@pytest.fixture
def cfg_dynamic(project_root, dyn_bundle) -> Config:
    return Config.load(
        project_root / "configs" / "synthetic_dynamic.yaml",
        overrides={
            "data": {"input_path": str(dyn_bundle["data"])},
            "tuning": {"enabled": False},
        },
    )


@pytest.fixture
def cfg_static(project_root, static_bundle) -> Config:
    return Config.load(
        project_root / "configs" / "synthetic_static.yaml",
        overrides={
            "data": {"input_path": str(static_bundle["data"])},
            "tuning": {"enabled": False},
        },
    )


@pytest.fixture
def dataset_dynamic(cfg_dynamic):
    return load_dataset(cfg_dynamic)


@pytest.fixture
def dataset_static(cfg_static):
    return load_dataset(cfg_static)
