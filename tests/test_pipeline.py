"""流水线级测试：目前只覆盖不依赖训练的小件。"""

from __future__ import annotations

import pytest

from floodrisk.pipeline import _assert_label_horizon_verified, _provenance_badge, _window_unit_warnings


def test_provenance_badge_only_real_is_citable():
    citable, banner = _provenance_badge("REAL")
    assert citable is True
    assert banner == "REAL_DATA"


def test_provenance_badge_unknown_is_not_citable():
    """来源不明的数据绝不能默认成 REAL_DATA —— C 类红线。"""
    citable, banner = _provenance_badge("UNKNOWN")
    assert citable is False
    assert "PROVENANCE_UNKNOWN" in banner


def test_provenance_badge_synthetic():
    citable, banner = _provenance_badge("SYNTHETIC")
    assert citable is False
    assert "SYNTHETIC_DO_NOT_CITE" in banner


def test_real_looking_filename_without_proof_is_not_citable(cfg_dynamic, tmp_path):
    """端到端：真实风格文件名 + 无来源证明 → UNKNOWN，且绝不标记 citable。"""
    import pandas as pd

    from floodrisk.io import load_dataset

    df = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "timestamp": ["2025-01-01 00:00", "2025-01-01 01:00"] * 2,
            "rainfall": [1.0, 2.0, 3.0, 4.0],
            "flood_label": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "hk_flood_2020_2024.csv"   # 真实风格，文件名不含 synthetic，也无旁挂
    df.to_csv(path, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(path)
    cfg_dynamic._data["project"]["provenance_override"] = None

    ds = load_dataset(cfg_dynamic)
    assert ds.provenance == "UNKNOWN"

    citable, banner = _provenance_badge(ds.provenance)
    assert citable is False
    assert banner.startswith("PROVENANCE_UNKNOWN")


def test_unverified_label_horizon_blocks_real_training(cfg_dynamic, dataset_dynamic):
    """非合成数据 + 口径未验证 → 必须中止，不能告警后继续训练。"""
    cfg_dynamic._data["task"]["label_horizon_verified"] = False
    dataset_dynamic.provenance = "REAL"

    with pytest.raises(RuntimeError, match="标签窗口口径未验证"):
        _assert_label_horizon_verified(dataset_dynamic, cfg_dynamic)

    # 显式声明已确认后才放行
    cfg_dynamic._data["task"]["label_horizon_verified"] = True
    _assert_label_horizon_verified(dataset_dynamic, cfg_dynamic)


def test_synthetic_data_is_not_blocked_by_horizon_gate(cfg_dynamic, dataset_dynamic):
    """合成标签由生成器按同一窗口构造，不受该闸门限制。"""
    cfg_dynamic._data["task"]["label_horizon_verified"] = False
    assert dataset_dynamic.provenance == "SYNTHETIC"
    _assert_label_horizon_verified(dataset_dynamic, cfg_dynamic)


def test_window_units_warn_when_time_step_changes(cfg_dynamic):
    """lags / rolling_windows 的单位是「步」：换时间粒度必须显式告警。"""
    assert _window_unit_warnings(cfg_dynamic) == []
    cfg_dynamic._data["task"]["time_step_hours"] = 4
    warns = _window_unit_warnings(cfg_dynamic)
    assert warns and warns[0]["code"] == "window_units_in_steps"
    assert "4" in warns[0]["message"]
