"""泄漏闸门测试。这是整套框架最该被写测试的地方。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodrisk.io import Dataset
from floodrisk.leakage import (
    LeakageError,
    assert_split_disjoint,
    audit_features,
    check_feature_names,
    check_time_lag,
    check_time_rolling,
    univariate_auc_scan,
)


# ---------------------------------------------------------------- 闸门 1
def test_leaky_name_is_blocked(cfg_dynamic):
    with pytest.raises(LeakageError, match="标签期信息"):
        check_feature_names(["rainfall", "flood_report_count"], cfg_dynamic)


def test_post_event_name_is_blocked(cfg_dynamic):
    with pytest.raises(LeakageError):
        check_feature_names(["drainage_clearance_records"], cfg_dynamic)


def test_legit_design_capacity_passes(cfg_dynamic):
    # "排水设计标准"是规划参数，不是事后记录，应放行
    assert check_feature_names(["rainfall", "elevation", "drainage_capacity"], cfg_dynamic) == []


def test_explicit_blacklist_works(cfg_dynamic):
    cfg_dynamic._data["leakage"]["forbidden_exact"] = ["my_proxy"]
    with pytest.raises(LeakageError, match="forbidden_exact"):
        check_feature_names(["my_proxy"], cfg_dynamic)


def test_target_itself_is_blocked(cfg_dynamic):
    with pytest.raises(LeakageError, match="标签列同名"):
        check_feature_names(["flood_label"], cfg_dynamic)


# ---------------------------------------------------------------- 闸门 2
def _panel_with_gap():
    """单网格 10 小时序列，第 5 小时缺报 —— 按行位移与按时间位移从这里开始分叉。"""
    times = pd.date_range("2025-01-01", periods=10, freq="h").delete(5)
    values = np.arange(len(times), dtype=float)
    df = pd.DataFrame({"grid_id": "G1", "timestamp": times, "rainfall": values})
    return df


def test_time_lag_passes_for_time_built_feature():
    df = _panel_with_gap()
    lookup = dict(zip(df["timestamp"], df["rainfall"]))
    df["rain_lag2"] = [lookup.get(t - pd.Timedelta(hours=2), np.nan) for t in df["timestamp"]]
    check_time_lag(
        df, grid_col="grid_id", time_col="timestamp",
        base_col="rainfall", feat_col="rain_lag2",
        lag_steps=2, min_lag=1, step_hours=1, name="rain_lag2",
    )


def test_row_shifted_feature_is_caught_despite_gap():
    """用 df.shift() 按行建的特征：缺口之后错位一小时，必须被数值校验抓出。

    这是"按时间位移 vs 按行位移"的核心场景 —— 没有时间缺口的合成数据
    永远测不出这个 bug。
    """
    df = _panel_with_gap()
    df["rain_lag2"] = df["rainfall"].shift(2)  # 按行位移
    with pytest.raises(LeakageError, match="按时间位移"):
        check_time_lag(
            df, grid_col="grid_id", time_col="timestamp",
            base_col="rainfall", feat_col="rain_lag2",
            lag_steps=2, min_lag=1, step_hours=1, name="rain_lag2",
        )


def test_lag_below_min_is_rejected():
    df = _panel_with_gap()
    with pytest.raises(LeakageError, match="最小合法滞后"):
        check_time_lag(
            df, grid_col="grid_id", time_col="timestamp",
            base_col="rainfall", feat_col="rainfall",
            lag_steps=0, min_lag=1, step_hours=1, name="rain_now",
        )


def test_unalignable_feature_raises():
    """没有任何可对齐样本时必须报错，不能默默通过。"""
    df = _panel_with_gap()
    df["junk"] = np.nan
    with pytest.raises(LeakageError, match="无法验证"):
        check_time_lag(
            df, grid_col="grid_id", time_col="timestamp",
            base_col="rainfall", feat_col="junk",
            lag_steps=1, min_lag=1, step_hours=1, name="junk",
        )


def test_rolling_time_built_passes():
    df = _panel_with_gap()
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    base_full = df.set_index("timestamp")["rainfall"].reindex(full)
    rolled = base_full.shift(1).rolling(2).mean()
    df = df.set_index("timestamp")
    df["rain_r2"] = rolled
    df = df.reset_index()
    check_time_rolling(
        df, grid_col="grid_id", time_col="timestamp",
        base_col="rainfall", feat_col="rain_r2",
        lag_steps=1, window=2, stat="mean", min_lag=1, step_hours=1, name="rain_r2",
    )


def test_rolling_row_built_is_caught():
    df = _panel_with_gap()
    df["rain_r2"] = df["rainfall"].shift(1).rolling(2).mean()  # 按行 → 缺口后错位
    with pytest.raises(LeakageError, match="不符"):
        check_time_rolling(
            df, grid_col="grid_id", time_col="timestamp",
            base_col="rainfall", feat_col="rain_r2",
            lag_steps=1, window=2, stat="mean", min_lag=1, step_hours=1, name="rain_r2",
        )


def test_duplicate_timestamp_raises():
    df = _panel_with_gap()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError, match="重复"):
        check_time_lag(
            df, grid_col="grid_id", time_col="timestamp",
            base_col="rainfall", feat_col="rainfall",
            lag_steps=1, min_lag=1, step_hours=1, name="x",
        )


# ---------------------------------------------------------------- 闸门 3
def test_row_overlap_detected():
    a = np.arange(0, 10)
    b = np.arange(5, 15)
    with pytest.raises(LeakageError, match="切分重叠"):
        assert_split_disjoint(a, b, np.arange(20, 25))


def test_block_leakage_detected():
    blocks = np.array(["A"] * 5 + ["B"] * 5)
    with pytest.raises(LeakageError, match="空间块泄漏"):
        assert_split_disjoint(np.arange(0, 5), np.arange(5, 8), np.arange(8, 10), blocks=blocks)


def test_clean_split_passes():
    blocks = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    assert_split_disjoint(
        np.arange(0, 5), np.arange(5, 10), np.arange(10, 15), blocks=blocks
    ) is None


# ---------------------------------------------------------------- 单变量 AUC 扫描
def test_univariate_scan_flags_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1] * 10)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "clean_noise": rng.normal(size=len(y)),
            "sneaky_perfect": y + rng.normal(0, 0.01, size=len(y)),
            "inverse_perfect": -y.astype(float),
        }
    )
    scan = univariate_auc_scan(X, y, warn_threshold=0.97)
    flagged = {r["feature"] for r in scan if r["suspect_leakage"]}
    assert "sneaky_perfect" in flagged
    assert "inverse_perfect" in flagged
    assert "clean_noise" not in flagged
    # 排序检查：可疑项应排在前列
    assert scan[0]["univariate_auc"] >= scan[-1]["univariate_auc"]


def test_univariate_scan_handles_nan():
    y = np.array([0, 1] * 20)
    X = pd.DataFrame({"has_nan": np.where(np.arange(40) % 7 == 0, np.nan, np.arange(40.0))})
    scan = univariate_auc_scan(X, y)
    assert len(scan) == 1


# ---------------------------------------------------------------- 汇总审计
def test_train_scoped_auc_scan_is_invariant_to_test_labels(dataset_dynamic, cfg_dynamic):
    """M3：训练集保持不变、只翻转测试集标签时，训练阶段的单变量 AUC 审计结果必须不变。

    若审计扫描了全量数据，测试标签会经审计报告回流到特征取舍决策 —— 这本身
    就是泄漏。修复后扫描被强制限制在训练段行内，因此对测试标签的任何扰动
    都不应改变审计结果。
    """
    from floodrisk.splits import make_split_plan

    plan = make_split_plan(dataset_dynamic, cfg_dynamic)

    audit_before = audit_features(dataset_dynamic, cfg_dynamic, scan_idx=plan.train_idx)
    auc_before = [r["univariate_auc"] for r in audit_before["univariate_auc"]]

    # 只翻转测试集标签；temporal 切分与标签无关，plan 行索引保持不变
    df_mod = dataset_dynamic.df.copy()
    y = df_mod["flood_label"].to_numpy(copy=True)
    y[plan.test_idx] = 1 - y[plan.test_idx]
    df_mod["flood_label"] = y
    ds_mod = Dataset(
        df=df_mod,
        features=dataset_dynamic.features,
        target=dataset_dynamic.target,
        provenance=dataset_dynamic.provenance,
        columns=dataset_dynamic.columns,
    )

    audit_after = audit_features(ds_mod, cfg_dynamic, scan_idx=plan.train_idx)
    auc_after = [r["univariate_auc"] for r in audit_after["univariate_auc"]]

    assert auc_before == auc_after
    assert "仅传入" in audit_after["auc_scan_scope"]
