"""校准层测试：校准必须真的把概率拉回来，且不能偷偷用测试集拟合。"""

from __future__ import annotations

import numpy as np
import pytest

from floodrisk.calibrate import (
    Calibrator,
    calibration_report,
    calibration_table,
    expected_calibration_error,
    fit_calibrator,
)


def _miscalibrated(n=4000, seed=0):
    """构造一个"排序对、但概率整体偏高"的分数 —— scale_pos_weight 的典型后果。"""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # 分数随标签单调上升，但被整体上推
    p = np.clip(0.55 * y + 0.35 + rng.normal(0, 0.12, n), 0.01, 0.99)
    return y, p


def test_identity_calibrator_leaves_scores_untouched():
    cal = fit_calibrator(np.array([0, 1]), np.array([0.2, 0.8]), method="none")
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal.transform(p), p)
    assert "未校准" in cal.describe()["note"]


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_reduces_brier(method):
    y, p = _miscalibrated()
    cal = fit_calibrator(y, p, method=method, min_samples=100)
    assert cal.method == method
    before = cal.info["brier_before"]
    after = cal.info["brier_after"]
    assert after < before


def test_calibrated_output_stays_in_unit_interval():
    y, p = _miscalibrated()
    cal = fit_calibrator(y, p, method="isotonic", min_samples=100)
    out = cal.transform(np.array([0.0, 0.001, 0.5, 0.999, 1.0]))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_platt_preserves_ranking_monotonically():
    """Platt 只是单调变换：不同分数之间的相对次序不得改变（并列值除外）。"""
    y, p = _miscalibrated()
    cal = fit_calibrator(y, p, method="platt", min_samples=100)
    uniq = np.unique(p)
    out = cal.transform(uniq)
    assert np.all(np.diff(out) >= -1e-12)
    assert out[-1] > out[0]


def test_small_validation_set_falls_back():
    y = np.array([0, 1] * 20)
    p = np.linspace(0.05, 0.95, 40)
    cal = fit_calibrator(y, p, method="isotonic", min_samples=200)
    assert cal.method == "none"
    assert cal.info["fallback"] is True


def test_ece_is_lower_after_calibration():
    y, p = _miscalibrated()
    cal = fit_calibrator(y, p, method="isotonic", min_samples=100)
    before = expected_calibration_error(y, p)
    after = expected_calibration_error(y, cal.transform(p))
    assert after < before


def test_calibration_table_rows_cover_all_bins():
    y, p = _miscalibrated(500)
    rows = calibration_table(y, p, n_bins=5)
    assert len(rows) == 5
    assert sum(r["n"] for r in rows) == len(y)


def test_report_records_fit_source_and_never_touches_test():
    y_fit, p_fit = _miscalibrated(3000, seed=1)
    cal = fit_calibrator(y_fit, p_fit, method="platt", min_samples=100)
    y_test, p_test = _miscalibrated(1000, seed=2)
    report = calibration_report(y_test, p_test, cal)
    assert report["details"]["fit_on"] == "val"
    assert report["details"]["n_fit"] == len(y_fit)
    assert "不参与任何参数估计" in report["note"]
    assert set(report) >= {"before", "after", "reliability_table"}


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="校准方法"):
        fit_calibrator(np.array([0, 1]), np.array([0.2, 0.8]), method="magic")
    with pytest.raises(ValueError):
        Calibrator("magic").transform(np.array([0.5]))
