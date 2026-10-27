"""超参搜索。

为什么不用 ``RandomizedSearchCV`` 直接跑
--------------------------------------
因为它的默认 CV 是 KFold —— 会把同一空间块、相邻时段切到训练与验证两侧，
前三道泄漏闸门全部失效，搜索出来的"最优参数"只是最会背答案的那一组。
而且我们的切分返回的是**全表行位置索引**，与 ``RandomizedSearchCV`` 期望的
"0..n-1 位置索引"不是同一套坐标系，直接塞进去会静默错位。

所以这里自己写一个小的随机搜索：
- 折必须由 ``splits.cv_splits`` 提供（与主切分同源，已含 purge 与空间分组）；
- 目标函数 = 各折指标均值，同时记录标准差（稳定性）与逐折明细；
- 只报告，不自动改配置 —— 选哪组参数仍然要人确认。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import binary_metrics
from .train import TrainError, build_model, predict_proba, resolve_imbalance


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    table: pd.DataFrame
    metric: str
    n_iter: int
    folds: int
    baseline_params: dict[str, Any] | None = None
    note: str = ""
    trials: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n_iter": self.n_iter,
            "folds": self.folds,
            "best_params": self.best_params,
            "best_score": float(self.table["mean_score"].max()) if len(self.table) else None,
            "note": self.note,
        }


def _metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "average_precision":
        from sklearn.metrics import average_precision_score

        return lambda y, p: float(average_precision_score(y, p))
    if name == "roc_auc":
        from sklearn.metrics import roc_auc_score

        return lambda y, p: float(roc_auc_score(y, p))
    if name == "f1":
        return lambda y, p: float(binary_metrics(y, p, threshold=0.5)["f1"])
    if name == "recall":
        return lambda y, p: float(binary_metrics(y, p, threshold=0.5)["recall"])
    raise TrainError(f"tuning.metric 不支持：{name!r}")


def random_search(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    *,
    n_iter: int | None = None,
    seed: int | None = None,
    log: Callable[[str], None] | None = None,
) -> TuningResult:
    """在给定折上做随机搜索。``folds`` 中的索引必须是 ``X`` 的位置索引。"""
    space = cfg.get("tuning.search_space", {}) or {}
    if not space:
        raise TrainError("tuning.search_space 为空，无法搜索")
    metric_name = cfg.get("tuning.metric", "average_precision")
    score = _metric_fn(metric_name)
    n_iter = int(n_iter if n_iter is not None else cfg.get("tuning.n_iter", 20))
    if n_iter < 1:
        raise TrainError("tuning.n_iter 必须 >= 1")
    if len(folds) < 2:
        raise TrainError(f"折数仅 {len(folds)}，超参搜索至少要 2 折才有意义")
    if len(np.unique(y)) < 2:
        raise TrainError("标签只有单一类别，无法调参")

    rng = np.random.default_rng(int(seed if seed is not None else cfg.get("split.seed", 42)))
    base_params = dict(cfg.get("train.params", {}) or {})
    imb_params, _ = resolve_imbalance(y, cfg)

    # 基线：配置里那组参数，作为"调参到底有没有用"的对照
    baseline = evaluate_params(base_params, X, y, folds, cfg, score, imb_params)

    rows: list[dict[str, Any]] = []
    for i in range(n_iter):
        trial = dict(base_params)
        for key, values in space.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise TrainError(f"tuning.search_space.{key} 必须是非空列表")
            trial[key] = values[int(rng.integers(0, len(values)))]
        try:
            res = evaluate_params(trial, X, y, folds, cfg, score, imb_params)
        except Exception as exc:  # 非法组合不该中断整个搜索
            rows.append({"trial": i, "mean_score": float("nan"), "std_score": float("nan"),
                         "error": str(exc)[:200], **{k: trial[k] for k in space}})
            continue
        rows.append({"trial": i, **{k: trial[k] for k in space}, **res})
        if log:
            log(f"  trial {i + 1:02d}/{n_iter}: {metric_name}={res['mean_score']:.4f} "
                f"± {res['std_score']:.4f}  {({k: trial[k] for k in space})}")

    table = pd.DataFrame(rows).sort_values("mean_score", ascending=False, na_position="last")
    best = table.iloc[0]
    best_params = {k: best[k] for k in space if k in best and pd.notna(best[k])}
    # 数值型搜索空间取回 Python 原生类型，避免 numpy 标量写进 yaml/json 出问题
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}

    improvement = (
        round(float(table["mean_score"].max()) - baseline["mean_score"], 6)
        if len(table) and pd.notna(table["mean_score"].max())
        else None
    )
    return TuningResult(
        best_params=best_params,
        table=table.reset_index(drop=True),
        metric=metric_name,
        n_iter=n_iter,
        folds=len(folds),
        baseline_params=baseline,
        note=(
            f"折由 splits.cv_splits 提供（与主切分同源，含 purge）；相对未调参基线提升 {improvement}"
            if improvement is not None
            else "折由 splits.cv_splits 提供；基线对比失败"
        ),
    )


def evaluate_params(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    score: Callable[[np.ndarray, np.ndarray], float],
    imb_params: dict[str, Any],
) -> dict[str, Any]:
    """一组参数在各折上的表现。"""
    merged = dict(params)
    merged.update(imb_params)
    merged.pop("early_stopping_rounds", None)  # 搜索时统一由 build_model 按配置加
    scores: list[float] = []
    for tr, va in folds:
        model = build_model(cfg, merged)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        best_iter = getattr(model, "best_iteration", None)
        p = predict_proba(
            model, X.iloc[va], best_iteration=None if (best_iter is None or best_iter < 0) else int(best_iter)
        )
        scores.append(score(y[va], p))
    arr = np.asarray(scores, dtype=float)
    return {
        "mean_score": round(float(arr.mean()), 6),
        "std_score": round(float(arr.std(ddof=0)), 6),
        "min_score": round(float(arr.min()), 6),
        "max_score": round(float(arr.max()), 6),
        "fold_scores": [round(float(s), 6) for s in scores],
    }
