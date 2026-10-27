"""概率校准。

为什么必须单独一层
------------------
XGBoost 输出的分数是"相对风险排序"，不是"发生概率"。用了
``scale_pos_weight`` 做过不平衡加权之后，分数会被系统性推高（模型先验被改过），
直接拿 0.6 当"60% 会水浸"是错的。

而三级风险分级恰恰要用概率的**绝对值**（分位切点之后还要对外解释
"高风险意味着什么"），所以校准不是可选装饰：

- ``platt``    —— 在验证集上拟合一个 logistic 回归，把分数拉回概率尺度。
  样本少时更稳（只估 2 个参数），但只能纠正在位单调映射。
- ``isotonic`` —— 保序回归，能纠正任意单调形变，但验证集少于几百个正例时
  会过拟合出阶梯状的怪曲线。

**校准器只在验证集上拟合，且绝不接触测试集** —— 否则测试指标就是自欺。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    """统一的校准器接口：fit(p, y) / transform(p)。"""

    method: str
    model: Any = None
    info: dict[str, Any] | None = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return p
        if self.method == "platt":
            z = _logit(p).reshape(-1, 1)
            return self.model.predict_proba(z)[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 0.0, 1.0)
        raise ValueError(f"未知校准方法：{self.method}")

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, **(self.info or {})}


def fit_calibrator(
    y_val: np.ndarray,
    p_val: np.ndarray,
    method: str = "none",
    *,
    min_samples: int = 200,
) -> Calibrator:
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    if method not in {"none", "platt", "isotonic"}:
        raise ValueError(f"校准方法只支持 none / platt / isotonic，当前 {method!r}")

    if method == "none":
        return Calibrator("none", None, {"note": "未校准：分数只可用于排序，不可解释为概率"})

    if len(y) < min_samples:
        return Calibrator(
            "none",
            None,
            {
                "note": f"验证集仅 {len(y)} 行 < {min_samples}，校准不可靠，已回退为 none",
                "requested": method,
                "fallback": True,
            },
        )
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return Calibrator(
            "none",
            None,
            {
                "note": f"验证集正例 {int(y.sum())} / 负例 {int(len(y) - y.sum())} 过少，校准不可靠，已回退",
                "requested": method,
                "fallback": True,
            },
        )

    before = float(brier_score_loss(y, p))
    if method == "platt":
        z = _logit(p).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        cal = Calibrator("platt", lr)
        cal.info = {
            "brier_before": round(before, 6),
            "brier_after": round(float(brier_score_loss(y, cal.transform(p))), 6),
            "coef": round(float(lr.coef_[0][0]), 4),
            "intercept": round(float(lr.intercept_[0]), 4),
            "fit_on": "val",
            "n_fit": int(len(y)),
        }
    else:  # isotonic
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        cal = Calibrator("isotonic", iso)
        cal.info = {
            "brier_before": round(before, 6),
            "brier_after": round(float(brier_score_loss(y, cal.transform(p))), 6),
            "fit_on": "val",
            "n_fit": int(len(y)),
            "unique_knots": int(len(np.unique(p))),
        }
    return cal


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """可靠性曲线数据：每个概率箱的"预测均值 vs 实际发生率"。"""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": n,
                "mean_predicted": round(float(p[mask].mean()), 6) if n else None,
                "observed_rate": round(float(y[mask].mean()), 6) if n else None,
                "gap": (
                    round(float(y[mask].mean() - p[mask].mean()), 6) if n else None
                ),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE：按样本量加权的 |预测 - 实际| 均值。0.05 以内算可接受。"""
    rows = calibration_table(y, p, n_bins)
    total = sum(r["n"] for r in rows)
    if not total:
        return float("nan")
    return round(
        sum(r["n"] * abs(r["gap"]) for r in rows if r["gap"] is not None) / total, 6
    )


def calibration_report(
    y: np.ndarray, p_raw: np.ndarray, calibrator: Calibrator, n_bins: int = 10
) -> dict[str, Any]:
    """校准前后对照。测试集只在这里被"应用"，不参与拟合。"""
    p_cal = calibrator.transform(p_raw)
    return {
        "method": calibrator.method,
        "details": calibrator.describe(),
        "before": {
            "brier": round(float(brier_score_loss(y, p_raw)), 6),
            "ece": expected_calibration_error(y, p_raw, n_bins),
            "mean_predicted": round(float(np.mean(p_raw)), 6),
            "observed_rate": round(float(np.mean(y)), 6),
        },
        "after": {
            "brier": round(float(brier_score_loss(y, p_cal)), 6),
            "ece": expected_calibration_error(y, p_cal, n_bins),
            "mean_predicted": round(float(np.mean(p_cal)), 6),
            "observed_rate": round(float(np.mean(y)), 6),
        },
        "reliability_table": calibration_table(y, p_cal, n_bins),
        "note": "校准器只在验证集拟合；测试集仅被 transform，不参与任何参数估计",
    }
