"""特征层测试。

重点不是"能算出来"，而是：
- 滞后到底对齐到了哪个时刻（有缺口时最容易错）；
- 滚动是"先位移再滚"还是"先滚再位移"；
- 预报窗口覆盖的是 t..t+h-1，不是 t+1..t+h；
- 静态变量不许造滞后，实测雨量不许进未来窗口。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodrisk.features import (
    FeatureError,
    _future_window,
    audit_built_features,
    build_features,
)
from floodrisk.io import Dataset


def _tiny_dataset(cfg, rows):
    df = pd.DataFrame(rows)
    df["flood_label"] = np.arange(len(df)) % 2
    df["spatial_block_id"] = "Z0"
    return Dataset(
        df=df,
        features=[c for c in df.columns if c not in {"grid_id", "timestamp", "flood_label", "spatial_block_id"}],
        target="flood_label",
        provenance="SYNTHETIC",
        columns=None,
    )


@pytest.fixture
def gap_dataset(cfg_dynamic):
    """单网格 12 小时，第 5 小时缺报，rainfall 值等于小时序号。"""
    times = pd.date_range("2025-01-01", periods=12, freq="h").delete(5)
    rows = [
        {
            "grid_id": "G1",
            "timestamp": t,
            "rainfall": float(t.hour),
            "tide": 1.0 + 0.1 * t.hour,
            "wind_speed": 2.0,
            "elevation": 10.0,
            "slope": 3.0,
            "drainage_capacity": 40.0,
            "tide_ast": 1.0 + 0.5 * np.sin(t.hour),
        }
        for t in times
    ]
    cfg_dynamic._data["features"]["temporal"]["lags"] = [1, 2, 3]
    cfg_dynamic._data["features"]["temporal"]["rolling_windows"] = [3]
    return cfg_dynamic, _tiny_dataset(cfg_dynamic, rows)


def test_lag_reads_the_right_hour_despite_gap(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    got = fs.X["rainfall_lag2"]
    # t=06:00 的前两小时是 04:00（值 4）——若按行位移会错取 03:00（值 3）
    t6 = ds.df.index[ds.df["timestamp"] == pd.Timestamp("2025-01-01 06:00")][0]
    assert got.loc[t6] == 4.0


def test_lag_into_gap_is_nan_not_backfilled(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    # t=06:00 回看 1 小时是 05:00，该小时缺报 → 必须 NaN，不能拿 04:00 顶上
    t6 = ds.df.index[ds.df["timestamp"] == pd.Timestamp("2025-01-01 06:00")][0]
    assert np.isnan(fs.X.loc[t6, "rainfall_lag1"])


def test_rolling_window_covers_past_only(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    t7 = ds.df.index[ds.df["timestamp"] == pd.Timestamp("2025-01-01 07:00")][0]
    # min_lag=1 位移后窗口 = t7-1, t7-2, t7-3 = 06, 05(缺), 04 → 窗口不完整 → NaN
    assert np.isnan(fs.X.loc[t7, "rainfall_r3mean"])
    # t8: 07, 06, 05(缺) → 仍不完整
    t8 = ds.df.index[ds.df["timestamp"] == pd.Timestamp("2025-01-01 08:00")][0]
    assert np.isnan(fs.X.loc[t8, "rainfall_r3mean"])


def test_static_variable_gets_no_lag_columns(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    assert "elevation" in fs.X.columns
    assert not [c for c in fs.X.columns if c.startswith("elevation_lag")]
    assert "elevation" in fs.families["static"]


def test_raw_observation_excluded_when_min_lag_is_one(gap_dataset):
    """保守策略下 lag 0 不合法：原始当期观测不得直接入模。"""
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    assert "rainfall" not in fs.X.columns
    assert fs.families["raw_observation"] == []


def test_raw_observation_included_when_min_lag_is_zero(gap_dataset):
    cfg, ds = gap_dataset
    cfg._data["features"]["temporal"]["min_lag_policy"] = "derived"
    cfg._data["features"]["temporal"]["lags"] = [0, 1]
    fs = build_features(ds, cfg)
    assert "rainfall" in fs.X.columns
    assert "rainfall" in fs.families["raw_observation"]


def test_future_window_covers_t_to_t_plus_h_minus_one():
    series = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    got = _future_window(series, 3, "max")
    # t0 的窗口是 [0,1,2] → 2；t2 的窗口是 [2,3,4] → 4；t3 起窗口越界 → NaN
    assert got.iloc[0] == 2.0
    assert got.iloc[2] == 4.0
    assert np.isnan(got.iloc[3])


def test_forecast_window_uses_future_tide_not_past(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    t0 = ds.df.index[ds.df["timestamp"] == pd.Timestamp("2025-01-01 00:00")][0]
    expect = max(ds.df.loc[t0 : t0 + 1, "tide_ast"])   # [t, t+2h) 共 2 步
    assert fs.X.loc[t0, "tide_ast_fw2max"] == pytest.approx(expect)


def test_observed_variable_cannot_enter_forecast_window(gap_dataset):
    """把实测雨量塞进未来窗口 = 直接把标签期信息喂给模型，必须报错。"""
    cfg, ds = gap_dataset
    cfg._data["features"]["forecast"]["variables"] = ["rainfall"]
    with pytest.raises(FeatureError, match="不是预报类变量"):
        build_features(ds, cfg)


def test_valid_mask_flags_rows_with_missing_history(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    # 序列开头历史不足、缺口前后、末行未来窗口越界 —— 一律标记为不可用，不填补
    assert fs.valid_mask.sum() < len(ds.df)
    assert not fs.valid_mask[-1]
    assert fs.report["unusable_rows"] == int((~fs.valid_mask).sum())
    # 被判为可用的行必须真的没有缺失
    assert fs.X.loc[fs.valid_mask].notna().all().all()


def test_audit_passes_on_generated_features(gap_dataset):
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    audit = audit_built_features(ds, fs, cfg)
    assert audit["lag_failures"] == []
    assert audit["rolling_failures"] == []
    assert audit["forecast_failures"] == []
    assert audit["rolling_features_checked"] > 0


def test_audit_catches_sign_flipped_lag(gap_dataset):
    """故意把 lag2 改成 lag1 的内容 —— 复核必须发现。"""
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    fs.X["rainfall_lag2"] = fs.X["rainfall_lag1"]
    audit = audit_built_features(ds, fs, cfg)
    assert audit["passed"] is False
    assert audit["lag_failures"]


def test_audit_catches_corrupted_rolling(gap_dataset):
    """故意用 lag1 的内容顶替 r3mean —— rolling 复核必须发现。"""
    cfg, ds = gap_dataset
    fs = build_features(ds, cfg)
    assert "rainfall_r3mean" in fs.X.columns
    fs.X["rainfall_r3mean"] = fs.X["rainfall_lag1"]
    audit = audit_built_features(ds, fs, cfg)
    assert audit["passed"] is False
    assert audit["rolling_failures"]


def test_dynamic_dataset_feature_matrix_is_aligned_and_finite(cfg_dynamic, dataset_dynamic):
    fs = build_features(dataset_dynamic, cfg_dynamic)
    assert len(fs.X) == len(dataset_dynamic.df)
    assert list(fs.X.index) == list(dataset_dynamic.df.index)
    usable = fs.X.loc[fs.valid_mask]
    assert usable.notna().all().all()
    assert fs.report["families"]["forecast_window"] == 2
