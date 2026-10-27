"""三模型统一封装：XGBoost / Random Forest / Logistic Regression。

为什么单独一个模块
------------------
现有 ``train.py`` 只支持 XGBoost，``predict.py`` 也从 ``run_dir/model.json``
按 XGBoost 反序列化。要做"三模型对比 + 独立加载"，必须有一个与算法无关的
保存/加载契约，否则每加一个算法就要改一次预测入口。

契约：一个 ``bundle`` = {backend, model, features, meta}，joblib 落盘。
加载后只依据 ``features`` 重排列顺序，缺列直接报错，不多不少。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKENDS = ("xgb", "rf", "lr")


@dataclass
class Bundle:
    backend: str
    model: Any
    features: list[str]
    meta: dict[str, Any] = field(default_factory=dict)


def build_model(backend: str, *, seed: int = 42, n_jobs: int = -1, params: dict | None = None):
    """构造未训练的模型。参数只给研发默认值，正式调参须单独批准。"""
    params = dict(params or {})
    if backend == "xgb":
        import xgboost as xgb

        base = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=n_jobs,
            random_state=seed,
        )
        base.update(params)
        return xgb.XGBClassifier(**base)

    if backend == "rf":
        from sklearn.ensemble import RandomForestClassifier

        base = dict(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
        base.update(params)
        return RandomForestClassifier(**base)

    if backend == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        base = dict(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs")
        base.update(params)
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(**base)),
            ]
        )

    raise ValueError(f"未知 backend {backend!r}，可选 {BACKENDS}")


def fit(backend: str, X: pd.DataFrame, y: np.ndarray, *, seed: int = 42, params: dict | None = None):
    model = build_model(backend, seed=seed, params=params)
    model.fit(X, y)
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """正类概率，统一转 float64。

    XGBoost 的 ``predict`` 返回 float32，直接把 float32 写进 CSV 只有 7 位有效数字，
    下游校准/复核会出现 1e-8 量级的"假差异"。这里统一成 float64。
    """
    p = model.predict_proba(X)
    return np.asarray(p[:, 1], dtype=np.float64)


def save_bundle(path: str | Path, bundle: Bundle) -> Path:
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    return p


def load_bundle(path: str | Path) -> Bundle:
    import joblib

    b = joblib.load(Path(path))
    if not isinstance(b, Bundle):
        raise TypeError(f"{path} 不是本模块保存的 Bundle：{type(b)}")
    return b


def predict_with_bundle(bundle: Bundle, rows: pd.DataFrame) -> np.ndarray:
    """独立预测入口：校验字段、按训练顺序重排、输出正类概率。"""
    missing = [c for c in bundle.features if c not in rows.columns]
    if missing:
        raise ValueError(f"预测输入缺少 {len(missing)} 个训练期特征：{missing[:8]}")
    extra = [c for c in rows.columns if c not in bundle.features]
    X = rows[bundle.features]
    p = predict_proba(bundle.model, X)
    if len(p) != len(rows):
        raise RuntimeError(f"预测行数不一致：输出 {len(p)} vs 输入 {len(rows)}")
    _ = extra  # 额外列被忽略，但保留变量便于调试
    return p
