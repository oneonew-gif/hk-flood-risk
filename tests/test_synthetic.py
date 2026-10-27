"""合成数据测试：确定性、来源标记、以及"标签率是人为设定的"这件事不被忘掉。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from floodrisk.io import detect_provenance
from floodrisk.leakage import LeakageError, check_feature_names
from floodrisk.synthetic import (
    generate_grids,
    generate_panel,
    generate_static,
    write_synthetic_bundle,
)


def test_panel_is_deterministic():
    a = generate_panel(12, 40, seed=99)
    b = generate_panel(12, 40, seed=99)
    c = generate_panel(12, 40, seed=100)
    assert a.equals(b)
    assert not a.equals(c)


def test_panel_shape_and_columns():
    df = generate_panel(9, 30, seed=1)
    # n_steps=30、窗口 2 步 → 每个网格可用发起时刻 29 个，再抽掉少量缺报行
    assert 9 * 28 <= len(df) < 9 * 29
    assert {
        "grid_id", "timestamp", "rainfall", "tide", "tide_ast", "wind_speed", "flood_label"
    } <= set(df.columns)
    assert df["grid_id"].nunique() == 9
    assert df["flood_label"].isin([0, 1]).all()


def test_label_window_tail_is_right_censored():
    """标签是 [t, t+2h) 的窗口标签：末端没有完整未来窗口，必须丢弃而不是硬凑。"""
    n_steps, w = 30, 2
    df = generate_panel(4, n_steps, seed=11)
    t0 = df["timestamp"].min()
    last_allowed = t0 + pd.Timedelta(hours=n_steps - w)
    assert df["timestamp"].max() <= last_allowed
    # 被删失的时点不应出现在任何网格上
    censored = {t0 + pd.Timedelta(hours=h) for h in range(n_steps - w + 1, n_steps)}
    assert not censored & set(df["timestamp"].unique())


def test_gaps_exist_so_row_and_time_shift_diverge():
    """注入的缺报是泄漏闸门的"测具"：没有缺口，按行与按时间位移永远一致。"""
    n_steps, w = 60, 2
    df = generate_panel(6, n_steps, seed=3)
    per_grid = df.groupby("grid_id")["timestamp"].nunique()
    assert len(df) < 6 * (n_steps - w + 1)
    assert (per_grid < n_steps - w + 1).any()


def test_label_rate_matches_target_roughly():
    df = generate_panel(40, 200, seed=5, target_rate=0.12)
    rate = df["flood_label"].mean()
    assert 0.05 < rate < 0.22, f"标签率 {rate:.3f} 偏离目标 0.12 太多"


def test_coordinates_are_nowhere_near_hong_kong():
    """虚构坐标必须在 (0°,0°) 附近，从几何上杜绝被误认成香港数据。"""
    grids = generate_grids(25, seed=3)
    assert grids["lon"].abs().max() < 1.0
    assert grids["lat"].abs().max() < 1.0


def test_provenance_is_embedded():
    df = generate_panel(6, 20, seed=2)
    assert df.attrs["provenance"] == "SYNTHETIC"


def test_leaky_column_is_blocked_by_guard():
    from floodrisk.config import Config
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(root / "configs" / "base.yaml")
    df = generate_panel(8, 30, seed=4, leaky_column=True)
    assert "flood_report_count" in df.columns
    with pytest.raises(LeakageError):
        check_feature_names(["flood_report_count"], cfg)


def test_dirty_variant_injects_problems():
    clean = generate_panel(10, 40, seed=6)
    dirty = generate_panel(10, 40, seed=6, dirty=True)
    assert dirty["rainfall"].isna().sum() > 0
    assert len(dirty) > len(clean)          # 注入了重复行
    assert (dirty["tide"] < -1.5).sum() > 0  # 注入了越界值


def test_static_generator_has_one_row_per_grid():
    df = generate_static(30, seed=8)
    assert len(df) == 30
    assert df["grid_id"].is_unique
    assert df["flood_label"].isin([0, 1]).all()


def test_bundle_writes_data_provenance_and_geometry(tmp_path):
    paths = write_synthetic_bundle(tmp_path, mode="dynamic", n_grids=6, n_steps=24, seed=2)
    assert paths["data"].exists()
    assert paths["provenance"].exists()
    assert paths["grid_layer"].exists()

    sidecar = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert sidecar["provenance"] == "SYNTHETIC"
    assert sidecar["citable"] is False
    assert detect_provenance(paths["data"]) == "SYNTHETIC"

    geo = json.loads(paths["grid_layer"].read_text(encoding="utf-8"))
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 6
    coords = geo["features"][0]["geometry"]["coordinates"][0][0]
    assert abs(coords[0]) < 1.0 and abs(coords[1]) < 1.0
    assert geo["features"][0]["properties"]["provenance"] == "SYNTHETIC"
