"""切分测试：核心是"绝不允许重叠"，以及"绝不偷偷退化成随机"。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodrisk.splits import (
    SplitError,
    SplitPlan,
    _assert_usable,
    assign_blocks,
    cv_splits,
    make_split_plan,
)


def _overlap(a, b) -> int:
    return len(set(a.tolist()) & set(b.tolist()))


def test_temporal_split_is_chronological(dataset_dynamic, cfg_dynamic):
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    df = dataset_dynamic.df
    ts = pd.to_datetime(df["timestamp"])

    assert plan.name == "temporal"
    assert _overlap(plan.train_idx, plan.val_idx) == 0
    assert _overlap(plan.train_idx, plan.test_idx) == 0
    assert _overlap(plan.val_idx, plan.test_idx) == 0

    train_max = ts.iloc[plan.train_idx].max()
    val_min, val_max = ts.iloc[plan.val_idx].min(), ts.iloc[plan.val_idx].max()
    test_min = ts.iloc[plan.test_idx].min()
    assert train_max < val_min
    assert val_max < test_min


def test_temporal_split_keeps_all_rows(dataset_dynamic, cfg_dynamic):
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    total = len(plan.train_idx) + len(plan.val_idx) + len(plan.test_idx)
    # purge 带内的行被主动剥离：不参与任何集合，但必须交代清楚
    assert total + plan.meta["purged_rows"] == len(dataset_dynamic.df)


def test_temporal_split_purges_label_window_overlap(dataset_dynamic, cfg_dynamic):
    """标签是 [t, t+h) 窗口：训练集末行的标签不得伸进验证段，否则两段共享事件。"""
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    ts = pd.to_datetime(dataset_dynamic.df["timestamp"])
    h = pd.Timedelta(hours=cfg_dynamic.horizon_steps)

    assert ts.iloc[plan.train_idx].max() + h <= ts.iloc[plan.val_idx].min()
    assert ts.iloc[plan.val_idx].max() + h <= ts.iloc[plan.test_idx].min()
    assert plan.meta["purged_rows"] > 0
    assert plan.meta["purge_hours"] == 2.0


def test_purge_can_be_turned_off(dataset_dynamic, cfg_dynamic):
    cfg_dynamic._data["split"]["purge_horizon"] = False
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    total = len(plan.train_idx) + len(plan.val_idx) + len(plan.test_idx)
    assert total == len(dataset_dynamic.df)
    assert plan.meta["purged_rows"] == 0


def test_spatial_block_split_has_disjoint_blocks(dataset_dynamic, cfg_dynamic):
    cfg_dynamic._data["split"]["strategy"] = "spatial_block"
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    blocks = assign_blocks(dataset_dynamic.df, cfg_dynamic)
    for a, b in (
        (plan.train_idx, plan.val_idx),
        (plan.train_idx, plan.test_idx),
        (plan.val_idx, plan.test_idx),
    ):
        assert set(np.unique(blocks[a])) & set(np.unique(blocks[b])) == set()


def test_spacetime_split_is_strictest(dataset_dynamic, cfg_dynamic):
    cfg_dynamic._data["split"]["strategy"] = "spacetime"
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    df = dataset_dynamic.df
    ts = pd.to_datetime(df["timestamp"])
    blocks = assign_blocks(df, cfg_dynamic)

    assert _overlap(plan.train_idx, plan.test_idx) == 0
    assert set(np.unique(blocks[plan.train_idx])) & set(np.unique(blocks[plan.test_idx])) == set()
    assert ts.iloc[plan.train_idx].max() < ts.iloc[plan.test_idx].min()
    assert plan.meta["unused_rows"] > 0  # 测试块的历史时段不参与


def test_random_split_works_but_is_flagged(dataset_dynamic, cfg_dynamic):
    cfg_dynamic._data["split"]["strategy"] = "random"
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    assert plan.name == "random"
    assert _overlap(plan.train_idx, plan.test_idx) == 0


def test_block_split_falls_back_to_kmeans_without_block_column(dataset_dynamic, cfg_dynamic):
    df = dataset_dynamic.df.drop(columns=["spatial_block_id"])
    dataset_dynamic.df = df  # 模拟 A 未提供行政区编码
    cfg_dynamic._data["split"]["spatial_block_column"] = None
    blocks = assign_blocks(df, cfg_dynamic)
    assert len(np.unique(blocks)) == int(cfg_dynamic.get("split.spatial_blocks"))


def test_no_coordinates_raises_instead_of_random_fallback(dataset_dynamic, cfg_dynamic):
    dataset_dynamic.df = dataset_dynamic.df.drop(
        columns=["spatial_block_id", "x_km", "y_km", "lon", "lat"]
    )
    with pytest.raises(SplitError, match="无法构造空间切分"):
        assign_blocks(dataset_dynamic.df, cfg_dynamic)


def test_cv_folds_temporal_are_disjoint(dataset_dynamic, cfg_dynamic):
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    folds = cv_splits(dataset_dynamic, cfg_dynamic, plan.train_idx)
    assert len(folds) >= 2
    for tr, va in folds:
        assert _overlap(tr, va) == 0
        assert len(va) > 0


def test_cv_folds_spatial_respect_blocks(dataset_dynamic, cfg_dynamic):
    cfg_dynamic._data["split"]["strategy"] = "spatial_block"
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    blocks = assign_blocks(dataset_dynamic.df, cfg_dynamic)
    folds = cv_splits(dataset_dynamic, cfg_dynamic, plan.train_idx)
    for tr, va in folds:
        assert set(np.unique(blocks[tr])) & set(np.unique(blocks[va])) == set()


def test_split_summary_has_label_stats(dataset_dynamic, cfg_dynamic):
    plan = make_split_plan(dataset_dynamic, cfg_dynamic)
    summary = plan.summary_with_labels(dataset_dynamic.y)
    for part in ("train", "val", "test"):
        assert summary[part]["n"] > 0
        assert 0 < summary[part]["positive_rate"] < 1


def test_val_single_class_is_rejected():
    """val 段单一类别会让早停/阈值/校准在下游以难懂的方式爆掉，必须在切分阶段拦住。"""
    y = np.array([0, 1] * 5 + [1] * 5 + [0, 1] * 5)
    plan = SplitPlan(
        "random",
        train_idx=np.arange(0, 10),
        val_idx=np.arange(10, 15),
        test_idx=np.arange(15, 25),
    )
    with pytest.raises(SplitError, match="val 只有单一类别"):
        _assert_usable(plan, y)


def test_val_empty_is_rejected():
    y = np.array([0, 1] * 10)
    plan = SplitPlan(
        "random",
        train_idx=np.arange(0, 10),
        val_idx=np.array([], dtype=int),
        test_idx=np.arange(10, 20),
    )
    with pytest.raises(SplitError, match="val 为空"):
        _assert_usable(plan, y)
