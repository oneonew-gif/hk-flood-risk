"""评价指标测试：指标算得对、退化情况不炸、阈值真的在起作用。"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk.evaluate import (
    best_f1_threshold,
    binary_metrics,
    compare_splits,
    curve_data,
    format_metrics_table,
    threshold_sweep,
)


def test_perfect_prediction_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.9, 0.95])
    m = binary_metrics(y, p, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["fn"] == 0 and m["fp"] == 0


def test_all_negative_model_has_high_accuracy_but_zero_recall():
    """这正是不能用 Accuracy 当主指标的原因。"""
    y = np.array([0] * 95 + [1] * 5)
    p = np.full(100, 0.01)
    m = binary_metrics(y, p, threshold=0.5)
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["recall"] == 0.0
    assert m["pr_auc"] == pytest.approx(0.05)


def test_confusion_matrix_counts_add_up():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    m = binary_metrics(y, p, threshold=0.4)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(y)
    assert m["confusion_matrix"]["matrix"][0][0] == m["tn"]
    assert m["confusion_matrix"]["matrix"][1][1] == m["tp"]


def test_pr_auc_baseline_is_positive_rate():
    y = np.array([0] * 80 + [1] * 20)
    p = np.full(100, 0.3)
    m = binary_metrics(y, p, threshold=0.5)
    assert m["baseline_pr_auc"] == pytest.approx(0.2)
    assert m["pr_auc"] == pytest.approx(0.2, abs=0.01)


def test_single_class_returns_none_instead_of_crashing():
    y = np.zeros(50, dtype=int)
    m = binary_metrics(y, np.random.random(50))
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert "单一类别" in m["degenerate"]


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError, match="threshold"):
        binary_metrics([0, 1], [0.2, 0.8], threshold=1.5)


def test_threshold_changes_operating_point():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.45, 0.5, 0.9])
    low = binary_metrics(y, p, threshold=0.3)
    high = binary_metrics(y, p, threshold=0.7)
    assert low["recall"] > high["recall"]
    assert high["precision"] >= low["precision"]


def test_threshold_sweep_is_monotone_in_recall():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    sweep = threshold_sweep(y, p)
    recalls = sweep["recall"].to_numpy()
    assert np.all(np.diff(recalls) <= 1e-9)


def test_best_f1_threshold_is_within_range():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    best = best_f1_threshold(y, p)
    assert 0.0 <= best["threshold"] <= 1.0


def test_curve_data_shapes():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.6])
    c = curve_data(y, p)
    assert len(c["roc"]["fpr"]) == len(c["roc"]["tpr"])
    assert len(c["pr"]["precision"]) == len(c["pr"]["recall"])


def test_compare_and_format():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.4, 0.2])
    table = compare_splits({"val": (y, p), "test": (y, p)}, threshold=0.5)
    assert set(table) == {"val", "test"}
    text = format_metrics_table(table)
    assert "roc_auc" in text and "val" in text
