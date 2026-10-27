"""调参层测试：折的来源、搜索空间校验、以及"绝不用默认 KFold"。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodrisk.features import build_features
from floodrisk.splits import cv_splits, make_split_plan
from floodrisk.train import TrainError
from floodrisk.tuning import random_search


@pytest.fixture
def prepared(request):
    cfg = request.getfixturevalue("cfg_dynamic")
    ds = request.getfixturevalue("dataset_dynamic")
    fs = build_features(ds, cfg)
    plan = make_split_plan(ds, cfg)
    keep = fs.valid_mask
    tr = plan.train_idx[keep[plan.train_idx]]
    return cfg, ds, fs, tr


def test_random_search_returns_table_with_all_trials(prepared):
    cfg, ds, fs, tr = prepared
    cfg._data["tuning"]["search_space"] = {"max_depth": [3, 4]}
    cfg._data["tuning"]["metric"] = "average_precision"
    folds = cv_splits(ds, cfg, tr)
    res = random_search(fs.X, ds.y, folds, cfg, n_iter=3, seed=1)
    assert res.n_iter == 3
    assert len(res.table) == 3
    assert set(res.best_params) <= {"max_depth"}
    assert res.best_params["max_depth"] in {3, 4}
    assert res.folds == len(folds)
    assert res.table["mean_score"].notna().any()


def test_search_reports_fold_level_detail(prepared):
    cfg, ds, fs, tr = prepared
    cfg._data["tuning"]["search_space"] = {"max_depth": [3, 4]}
    folds = cv_splits(ds, cfg, tr)
    res = random_search(fs.X, ds.y, folds, cfg, n_iter=2, seed=2)
    row = res.table.iloc[0]
    assert len(row["fold_scores"]) == len(folds)
    assert row["min_score"] <= row["mean_score"] <= row["max_score"]


def test_empty_search_space_raises(prepared):
    cfg, ds, fs, tr = prepared
    cfg._data["tuning"]["search_space"] = {}
    folds = cv_splits(ds, cfg, tr)
    with pytest.raises(TrainError, match="search_space"):
        random_search(fs.X, ds.y, folds, cfg, n_iter=2)


def test_single_fold_raises(prepared):
    cfg, ds, fs, tr = prepared
    cfg._data["tuning"]["search_space"] = {"max_depth": [3]}
    folds = cv_splits(ds, cfg, tr)[:1]
    with pytest.raises(TrainError, match="至少要 2 折"):
        random_search(fs.X, ds.y, folds, cfg, n_iter=2)


def test_score_is_computed_on_validation_fold_only(prepared):
    """用打乱标签的数据调参：验证折上的得分必须掉到接近基线，说明没在训练折上打分。"""
    cfg, ds, fs, tr = prepared
    cfg._data["tuning"]["search_space"] = {"max_depth": [4]}
    folds = cv_splits(ds, cfg, tr)
    rng = np.random.default_rng(0)
    y_shuffled = rng.permutation(ds.y)
    res = random_search(fs.X, y_shuffled, folds, cfg, n_iter=1, seed=0)
    pr_auc = res.table["mean_score"].iloc[0]
    assert pr_auc < 0.6, "打乱标签后仍能拿到高分 → 打分用错了数据"
