"""模型评价：不平衡二分类该看的指标，一个都不能少，也一个都不能单看。

为什么不用 Accuracy 当主指标
---------------------------
水浸是稀有事件（真实数据正例率通常个位数百分比）。一个"全预测为不水浸"的
模型在 5% 正例率的数据上 Accuracy = 95%，但它一个水浸都报不出来 ——
对预警系统来说这是最坏的结果。所以：

- **主指标 PR-AUC（average precision）**：只看正例这一侧的排序质量，
  不受大量负例稀释；
- **ROC-AUC 并列报告**：便于与文献对比，但它对不平衡不敏感，单独用会高估；
- **Precision / Recall / F1 在指定阈值下**：预警系统真正部署的是阈值，
  不是概率排序，这三个数才是运行时的表现；
- **Brier / log-loss**：概率是否可信 —— 三级分级用的是概率绝对值，
  校准不好会让"高风险"档失去意义。

阈值不写死：``binary_metrics`` 必须显式传 threshold，三级分级的切点由
``tiering`` 模块在验证集上定，这里只提供扫描表供你挑。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(
    y_true: Iterable,
    y_prob: Iterable,
    *,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, Any]:
    """一次算全指标。单类别输入不抛错，而是把不可定义的指标标为 None 并说明原因。"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) != len(p):
        raise ValueError(f"标签与概率长度不一致：{len(y)} vs {len(p)}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold 必须在 [0,1]，当前 {threshold}")

    n = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out: dict[str, Any] = {
        "label": label,
        "n": n,
        "positives": n_pos,
        "negatives": n_neg,
        "positive_rate": round(n_pos / n, 6) if n else None,
        "threshold": float(threshold),
    }
    if n_pos == 0 or n_neg == 0:
        out["degenerate"] = "样本只有单一类别，排序类指标（AUC/PR-AUC）不可定义"
        for k in ("roc_auc", "pr_auc", "brier", "log_loss"):
            out[k] = None
        out["confusion_matrix"] = None
        return out

    out["roc_auc"] = round(float(roc_auc_score(y, p)), 6)
    out["pr_auc"] = round(float(average_precision_score(y, p)), 6)
    out["brier"] = round(float(brier_score_loss(y, p)), 6)
    out["log_loss"] = round(float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))), 6)
    out["baseline_pr_auc"] = round(n_pos / n, 6)
    """随机猜测的 PR-AUC 基线 = 正例率。PR-AUC 必须和它比，不能和 0.5 比"""
    out["pr_auc_lift"] = round(out["pr_auc"] / (n_pos / n), 3)

    y_hat = (p >= float(threshold)).astype(int)
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out.update(
        {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round((tp + tn) / n, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "specificity": round(tn / (tn + fp), 6) if (tn + fp) else None,
            "false_alarm_rate": round(fp / (fp + tn), 6) if (fp + tn) else None,
            "missing_rate": round(fn / (fn + tp), 6) if (fn + tp) else None,
            "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist()},
        }
    )
    return out


def threshold_sweep(
    y_true: Iterable,
    y_prob: Iterable,
    *,
    start: float = 0.01,
    stop: float = 0.99,
    step: float = 0.01,
) -> pd.DataFrame:
    """逐阈值扫描 P/R/F1，供你人工挑切点（或交给 tiering 做约束求解）。"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    thresholds = np.round(np.arange(start, stop + 1e-9, step), 6)
    rows = []
    for t in thresholds:
        m = binary_metrics(y, p, threshold=float(t))
        rows.append(
            {
                "threshold": float(t),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "tp": m.get("tp"),
                "fp": m.get("fp"),
                "fn": m.get("fn"),
                "tn": m.get("tn"),
            }
        )
    return pd.DataFrame(rows)


def curve_data(y_true: Iterable, y_prob: Iterable) -> dict[str, list[float]]:
    """ROC 与 PR 曲线坐标，用于画图（批 4 出图时直接用）。"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        "roc": {"fpr": fpr.round(6).tolist(), "tpr": tpr.round(6).tolist()},
        "pr": {"precision": prec.round(6).tolist(), "recall": rec.round(6).tolist()},
    }


def best_f1_threshold(y_true: Iterable, y_prob: Iterable) -> dict[str, float]:
    """仅作为参考点，不当作默认阈值 —— 切点应由业务或成本函数决定。"""
    sweep = threshold_sweep(y_true, y_prob)
    idx = sweep["f1"].idxmax()
    row = sweep.loc[idx]
    return {"threshold": float(row["threshold"]), "f1": float(row["f1"])}


def compare_splits(
    results: dict[str, tuple[Iterable, Iterable]],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """``{split: (y, p)}`` → ``{split: metrics}``，便于表格化对照。"""
    return {
        name: binary_metrics(y, p, threshold=threshold, label=name)
        for name, (y, p) in results.items()
    }


def format_metrics_table(metrics: dict[str, dict[str, Any]]) -> str:
    """把多组指标排成对齐的文本表，方便直接贴进日志。"""
    cols = ["n", "positives", "roc_auc", "pr_auc", "precision", "recall", "f1", "brier"]
    header = f"{'split':10s}" + "".join(f"{c:>12s}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in metrics.items():
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append(f"{'-':>12s}" if v is None else f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12d}")
        lines.append(f"{name:10s}" + "".join(cells))
    return "\n".join(lines)
