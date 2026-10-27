"""XGBoost 训练。

三件必须显式处理的事
--------------------
1. **类别不平衡**：水浸正例通常 1%-10%。默认的 logloss 会让模型倾向于"全报负例"，
   所以要么 ``scale_pos_weight = neg/pos``，要么调 ``max_delta_step``。
   本模块把选择权交给配置，并在 manifest 里记录用了哪个 —— 因为
   ``scale_pos_weight`` 会把输出概率系统性推高，必须靠校准层拉回来。

2. **早停用验证集，且只能看验证集**。测试集在训练全程不可见：早停轮数、
   超参、三级阈值都在验证集上定，最后才碰测试集一次。

3. **不做静默兜底**：训练集单类别、样本量过小、超参非法，一律报错而不是
   自动改一改继续跑。跑出来的数要能进材料，就不能有"其实降级了"的隐藏分支。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import binary_metrics


class TrainError(RuntimeError):
    """训练条件不成立。"""


@dataclass
class TrainResult:
    model: Any
    params: dict[str, Any]
    imbalance: dict[str, Any]
    best_iteration: int | None = None
    evals_result: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    val_metrics: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "imbalance": self.imbalance,
            "best_iteration": self.best_iteration,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


# ---------------------------------------------------------------------------
# 不平衡处理
# ---------------------------------------------------------------------------
def resolve_imbalance(y_train: np.ndarray, cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """把 ``train.imbalance`` 配置翻译成 XGBoost 参数，并记录它对概率含义的影响。"""
    strategy = cfg.get("train.imbalance.strategy", "none")
    y = np.asarray(y_train).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise TrainError(f"训练集只有单一类别（正例 {n_pos} / 负例 {n_neg}），无法训练")

    info: dict[str, Any] = {
        "strategy": strategy,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": round(n_pos / len(y), 6),
    }
    params: dict[str, Any] = {}

    if strategy == "none":
        info["note"] = "未做不平衡处理：模型会偏向多数类，Recall 可能很低"
    elif strategy == "scale_pos_weight":
        raw = cfg.get("train.imbalance.scale_pos_weight", "auto")
        weight = (n_neg / n_pos) if str(raw).lower() == "auto" else float(raw)
        if weight <= 0:
            raise TrainError(f"scale_pos_weight 必须 > 0，当前 {weight}")
        params["scale_pos_weight"] = float(weight)
        info["scale_pos_weight"] = float(weight)
        info["note"] = (
            "已用 scale_pos_weight 加权：输出分数被系统性推高，"
            "**必须经校准层才能解释为概率**"
        )
    elif strategy == "max_delta_step":
        step = float(cfg.get("train.imbalance.max_delta_step", 0))
        if step <= 0:
            raise TrainError("strategy=max_delta_step 时 train.imbalance.max_delta_step 必须 > 0")
        params["max_delta_step"] = step
        info["max_delta_step"] = step
        info["note"] = "用 max_delta_step 限制每轮叶子权重上限，概率尺度基本保持"
    else:  # pragma: no cover - config 已校验
        raise TrainError(f"未知的不平衡策略：{strategy}")
    return params, info


# ---------------------------------------------------------------------------
# 建模与训练
# ---------------------------------------------------------------------------
def build_model(cfg: Config, params: dict[str, Any] | None = None) -> Any:
    import xgboost as xgb

    base = dict(cfg.get("train.params", {}) or {})
    if params:
        base.update(params)
    base.setdefault("tree_method", "hist")
    base["random_state"] = int(cfg.get("train.random_state", 42))
    base["n_jobs"] = int(cfg.get("train.n_jobs", -1))
    base["eval_metric"] = cfg.get("train.eval_metric", "aucpr")

    esr = int(cfg.get("train.early_stopping_rounds", 0) or 0)
    if esr > 0:
        base["early_stopping_rounds"] = esr

    # 参数名拼错不会在构造时报错，只会在 fit 时报一个难懂的底层异常；
    # 这里提前对照 XGBoost 的默认参数集合检查一遍。
    known = set(xgb.XGBClassifier().get_params().keys())
    unknown = sorted(set(base) - known)
    if unknown:
        raise TrainError(
            f"train.params 中有 XGBoost 不认识的参数：{unknown}。"
            f"拼写检查一下，或确认当前 xgboost={xgb.__version__} 是否支持"
        )
    try:
        return xgb.XGBClassifier(**base)
    except TypeError as exc:  # pragma: no cover - 版本兼容
        raise TrainError(f"XGBoost 参数不被接受：{exc}。当前 xgboost={xgb.__version__}") from exc


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: Config,
    *,
    params: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> TrainResult:
    if len(X_train) < 30:
        raise TrainError(
            f"训练集仅 {len(X_train)} 行，样本量不足以训练。"
            f"静态模式请增加网格数；动态模式请扩大时间跨度"
        )
    if len(X_val) == 0:
        raise TrainError("验证集为空：早停与阈值确定都需要独立验证集")
    if list(X_train.columns) != list(X_val.columns):
        raise TrainError("训练集与验证集特征列不一致，检查特征构造是否稳定")

    imb_params, imb_info = resolve_imbalance(y_train, cfg)
    merged = dict(params or {})
    merged.update(imb_params)
    model = build_model(cfg, merged)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter < 0:
        best_iter = None
    p_val = predict_proba(model, X_val, best_iteration=best_iter)
    metrics = binary_metrics(y_val, p_val, threshold=threshold, label="val")

    evals = {}
    try:
        evals = {k: {m: [float(x) for x in v] for m, v in d.items()} for k, d in model.evals_result().items()}
    except Exception:  # pragma: no cover - 某些版本无 evals_result
        pass

    return TrainResult(
        model=model,
        params={k: v for k, v in model.get_params().items() if k != "kwargs"},
        imbalance=imb_info,
        best_iteration=int(best_iter) if best_iter is not None else None,
        evals_result=evals,
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        val_metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 特征重要性
# ---------------------------------------------------------------------------
def importance_table(
    model: Any,
    feature_names: list[str],
    *,
    kinds: tuple[str, ...] = ("gain", "weight", "cover"),
) -> pd.DataFrame:
    """Tree-based 重要性（gain / weight / cover），按 gain 归一化排序。"""
    booster = model.get_booster()
    frames = []
    for kind in kinds:
        if kind == "permutation":
            continue
        score = booster.get_score(importance_type=kind)
        col = pd.Series({f: float(score.get(f, 0.0)) for f in feature_names}, name=kind)
        frames.append(col)
    if not frames:
        raise TrainError("kinds 中至少要有一个 tree-based 重要性类型")
    df = pd.concat(frames, axis=1).fillna(0.0)
    total = df["gain"].sum()
    df["gain_share"] = (df["gain"] / total) if total > 0 else 0.0
    df.index.name = "feature"
    return df.sort_values("gain", ascending=False)


def permutation_importance_table(
    model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """置换重要性：打乱单列后 PR-AUC 掉多少。

    与 gain 的区别：gain 是训练集内的分裂收益（偏爱高基数特征），
    置换重要性是在**验证集**上测"这一列到底有没有用"，更能反映真实贡献。
    """
    from sklearn.inspection import permutation_importance as perm_imp

    res = perm_imp(
        model,
        X_val,
        y_val,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": X_val.columns,
            "perm_mean": res.importances_mean,
            "perm_std": res.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)


def predict_proba(
    model: Any,
    X: pd.DataFrame,
    *,
    best_iteration: int | None = None,
) -> np.ndarray:
    """统一预测入口，显式限定迭代轮数。

    早停之后"用第几棵树预测"在不同 xgboost 版本里默认行为不完全一致；
    这里显式传 ``iteration_range``，让"best_iteration 之后不再加树"这件事
    由我们自己保证，可复现、也便于写进方法说明。
    """
    if best_iteration is None or best_iteration < 0:
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X, iteration_range=(0, best_iteration + 1))[:, 1]


def constant_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    """"永远报负例"的基线。任何模型都必须明显强于它，否则等于没训练。"""
    rate = float(np.mean(y_train)) if len(y_train) else 0.0
    p = np.full(len(y_eval), rate)
    m = binary_metrics(y_eval, p, threshold=0.5, label="constant_baseline")
    m["note"] = "以训练集正例率作为常数预测，用于对照；PR-AUC 基线应等于正例率"
    return m
