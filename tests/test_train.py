"""训练层测试：不平衡处理、早停、重要性，以及"该报错的时候真的报错"。"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk.features import build_features
from floodrisk.splits import make_split_plan
from floodrisk.train import (
    TrainError,
    build_model,
    importance_table,
    predict_proba,
    resolve_imbalance,
    train_model,
)


@pytest.fixture
def prepared(request):
    """构造一次训练就绪的数据（夹具是函数级，因为 cfg 会被各用例改写）。"""
    cfg = request.getfixturevalue("cfg_dynamic")
    ds = request.getfixturevalue("dataset_dynamic")
    fs = build_features(ds, cfg)
    plan = make_split_plan(ds, cfg)
    keep = fs.valid_mask
    tr = plan.train_idx[keep[plan.train_idx]]
    va = plan.val_idx[keep[plan.val_idx]]
    te = plan.test_idx[keep[plan.test_idx]]
    return cfg, ds, fs, tr, va, te


def test_scale_pos_weight_auto_is_neg_over_pos(prepared):
    cfg, ds, fs, tr, va, te = prepared
    y = ds.y[tr]
    params, info = resolve_imbalance(y, cfg)
    expect = (y == 0).sum() / (y == 1).sum()
    assert params["scale_pos_weight"] == pytest.approx(expect)
    assert "校准" in info["note"]


def test_single_class_training_raises(cfg_dynamic):
    with pytest.raises(TrainError, match="单一类别"):
        resolve_imbalance(np.zeros(100, dtype=int), cfg_dynamic)


def test_max_delta_step_strategy_requires_positive(cfg_dynamic):
    cfg_dynamic._data["train"]["imbalance"] = {"strategy": "max_delta_step", "max_delta_step": 0}
    with pytest.raises(TrainError, match="max_delta_step"):
        resolve_imbalance(np.array([0, 1] * 50), cfg_dynamic)


def test_train_produces_usable_model(prepared):
    cfg, ds, fs, tr, va, te = prepared
    res = train_model(fs.X.iloc[tr], ds.y[tr], fs.X.iloc[va], ds.y[va], cfg)
    assert res.model is not None
    assert res.best_iteration is None or res.best_iteration >= 0
    assert res.val_metrics["roc_auc"] is not None
    # 必须明显强于随机：PR-AUC 至少高于正例率
    assert res.val_metrics["pr_auc"] > res.val_metrics["baseline_pr_auc"]


def test_tiny_training_set_rejected(prepared):
    cfg, ds, fs, tr, va, te = prepared
    with pytest.raises(TrainError, match="样本量不足"):
        train_model(fs.X.iloc[tr[:20]], ds.y[tr[:20]], fs.X.iloc[va], ds.y[va], cfg)


def test_empty_validation_set_rejected(prepared):
    cfg, ds, fs, tr, va, te = prepared
    with pytest.raises(TrainError, match="验证集为空"):
        train_model(fs.X.iloc[tr], ds.y[tr], fs.X.iloc[tr[:0]], ds.y[tr[:0]], cfg)


def test_feature_mismatch_rejected(prepared):
    cfg, ds, fs, tr, va, te = prepared
    with pytest.raises(TrainError, match="特征列不一致"):
        train_model(fs.X.iloc[tr], ds.y[tr], fs.X.iloc[va].drop(columns=[fs.names[0]]), ds.y[va], cfg)


def test_predict_proba_honours_iteration_range(prepared):
    cfg, ds, fs, tr, va, te = prepared
    res = train_model(fs.X.iloc[tr], ds.y[tr], fs.X.iloc[va], ds.y[va], cfg)
    if res.best_iteration is None or res.best_iteration <= 1:
        pytest.skip("本数据未触发早停，无法比较迭代区间")
    p_best = predict_proba(res.model, fs.X.iloc[va], best_iteration=res.best_iteration)
    p_first = predict_proba(res.model, fs.X.iloc[va], best_iteration=0)
    assert len(p_best) == len(fs.X.iloc[va])
    assert np.allclose(p_best, p_first) is False


def test_importance_shares_sum_to_one(prepared):
    cfg, ds, fs, tr, va, te = prepared
    res = train_model(fs.X.iloc[tr], ds.y[tr], fs.X.iloc[va], ds.y[va], cfg)
    imp = importance_table(res.model, fs.names)
    assert set(imp.columns) >= {"gain", "weight", "cover", "gain_share"}
    assert imp["gain_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert imp.index.name == "feature"


def test_build_model_rejects_unknown_param(cfg_dynamic):
    with pytest.raises(TrainError):
        build_model(cfg_dynamic, {"this_param_does_not_exist": 1})
