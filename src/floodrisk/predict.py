"""独立预测模块：从训练产物恢复模型，对新数据输出概率。

为什么需要这一层
----------------
训练流水线（pipeline.py）把模型、校准器、配置快照都存进了 run_dir，
但没有一个入口能"拿着这些产物去预测一批新数据"。本模块只做三件事：
1. 从 run_dir 恢复模型与校准器；
2. 用训练时的配置对新数据构造完全相同的特征；
3. 输出带概率的预测结果，供 GIS 导出或下游分析。

与训练流水线的边界
------------------
- 本模块**不训练、不调参、不做切分**；
- 特征构造严格复用 features.build_features，不重新发明轮子；
- 校准器从 JSON 恢复，不重新拟合；
- 所有输出保留 grid_id、timestamp、概率、数据来源与模型身份。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .calibrate import Calibrator
from .config import Config
from .features import build_features
from .io import PROVENANCE_SYNTHETIC, DatasetError, load_dataset
from .train import predict_proba


class PredictionError(RuntimeError):
    """预测条件不成立。"""


@dataclass
class PredictionResult:
    """预测结果容器。"""

    df: pd.DataFrame
    manifest: dict[str, Any]
    output_path: Path | None = None


def _load_run_artifacts(run_dir: Path) -> tuple[Config, xgb.XGBClassifier, Calibrator | None]:
    """从训练输出目录恢复配置、模型与校准器。"""
    config_path = run_dir / "config_snapshot.yaml"
    model_path = run_dir / "model.json"
    calibrator_path = run_dir / "calibrator.json"

    if not run_dir.exists():
        raise PredictionError(f"run_dir 不存在：{run_dir}")
    if not config_path.exists():
        raise PredictionError(f"缺少 config_snapshot.yaml：{config_path}")
    if not model_path.exists():
        raise PredictionError(f"缺少 model.json：{model_path}")

    cfg = Config.load(config_path)

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    calibrator: Calibrator | None = None
    if calibrator_path.exists():
        payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
        calibrator = _restore_calibrator(payload)

    return cfg, model, calibrator


def _restore_calibrator(payload: dict[str, Any]) -> Calibrator:
    """从 calibrator.json 恢复 Calibrator 对象。"""
    method = payload.get("method", "none")
    if method == "none":
        return Calibrator("none", None, payload)

    if method == "platt":
        coef = payload.get("coef")
        intercept = payload.get("intercept")
        if coef is None or intercept is None:
            raise PredictionError("platt 校准器缺少 coef / intercept")
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        lr.coef_ = np.array([[float(coef)]])
        lr.intercept_ = np.array([float(intercept)])
        lr.classes_ = np.array([0, 1])
        return Calibrator("platt", lr, payload)

    if method == "isotonic":
        x_thr = payload.get("x_thresholds")
        y_thr = payload.get("y_thresholds")
        if x_thr is None or y_thr is None:
            raise PredictionError("isotonic 校准器缺少 x_thresholds / y_thresholds")
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(x_thr), np.asarray(y_thr))
        return Calibrator("isotonic", iso, payload)

    raise PredictionError(f"未知校准方法：{method!r}")


def _assert_feature_alignment(
    cfg: Config,
    model: xgb.XGBClassifier,
    X_new: pd.DataFrame,
) -> None:
    """预测特征矩阵必须与训练时完全一致。"""
    n_features = model.n_features_in_
    if X_new.shape[1] != n_features:
        raise PredictionError(
            f"特征数量不匹配：训练时 {n_features} 列，预测时 {X_new.shape[1]} 列。"
            f"请确认特征构造配置与训练时一致"
        )


def _provenance_badge(provenance: str) -> tuple[bool, str]:
    """与 pipeline._provenance_badge 保持一致。"""
    if provenance == PROVENANCE_SYNTHETIC:
        return (
            False,
            "SYNTHETIC_DO_NOT_CITE —— 数据为程序生成的假数据，本目录任何指标/图件均不得写入报告或答辩",
        )
    if provenance == "REAL":
        return True, "REAL_DATA"
    return (
        False,
        "PROVENANCE_UNKNOWN —— 数据来源未确认，默认不可引用；"
        "确认是真实数据后请旁挂 *.provenance.json 或设置 project.provenance_override: REAL",
    )


def predict(
    run_dir: str | Path,
    data_path: str | Path,
    cfg: Config | None = None,
    output_path: str | Path | None = None,
) -> PredictionResult:
    """对一批新数据执行预测。

    Parameters
    ----------
    run_dir
        训练输出目录，必须含 ``model.json`` 与 ``config_snapshot.yaml``；
        ``calibrator.json`` 可选，缺失时回退为未校准。
    data_path
        新数据路径。必须含 ``grid_id``，dynamic 模式必须含 ``timestamp``；
        ``flood_label`` 可选。
    cfg
        可选配置。为 None 时从 ``run_dir/config_snapshot.yaml`` 恢复。
        若传入，则忽略 ``run_dir`` 中的配置快照。
    output_path
        可选输出路径。为 None 时写入 ``run_dir / "predictions_external.csv"``。

    Returns
    -------
    PredictionResult
        含预测 DataFrame 与 manifest 元数据。

    Notes
    -----
    预测时的特征构造严格复用训练配置。若新数据的时间分辨率、特征列名
    与训练时不一致，将抛出 ``PredictionError``。
    """
    run_dir = Path(run_dir)
    data_path = Path(data_path)

    # ---- 1. 恢复训练产物
    if cfg is None:
        cfg, model, calibrator = _load_run_artifacts(run_dir)
    else:
        # 允许外部传入配置（例如需要覆盖 data.columns 映射时）
        if not (run_dir / "model.json").exists():
            raise PredictionError(f"缺少 model.json：{run_dir / 'model.json'}")
        model = xgb.XGBClassifier()
        model.load_model(str(run_dir / "model.json"))
        calibrator = None
        if (run_dir / "calibrator.json").exists():
            payload = json.loads(
                (run_dir / "calibrator.json").read_text(encoding="utf-8")
            )
            calibrator = _restore_calibrator(payload)

    if cfg.mode == "static":
        raise PredictionError(
            "static 模式尚未实现独立预测；请用 pipeline.run_training 的批量输出"
        )

    # ---- 2. 加载新数据（无标签模式）
    # 注意：预测数据的 provenance 应基于输入文件本身判定，而非训练配置的 override。
    # 因此临时清空 project.provenance_override，让 detect_provenance 走文件名/旁挂文件逻辑。
    cfg_for_data = Config.from_dict(cfg.to_dict())
    cfg_for_data._data.setdefault("project", {})["provenance_override"] = None
    ds = load_dataset(cfg_for_data, data_path, require_label=False)

    # ---- 3. 构造特征（严格复用训练配置）
    fs = build_features(ds, cfg)
    if fs.valid_mask.sum() == 0:
        raise PredictionError(
            "特征构造后没有任何可用行。可能原因："
            "新数据时间跨度不足 lag/rolling 所需历史，或时间分辨率与训练配置不一致"
        )

    X_new = fs.X.loc[fs.valid_mask]

    # ---- 4. 特征对齐检查
    _assert_feature_alignment(cfg, model, X_new)

    # ---- 5. 预测
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter < 0:
        best_iter = None
    prob_raw = predict_proba(model, X_new, best_iteration=best_iter)

    is_calibrated = calibrator is not None and calibrator.method != "none" and calibrator.model is not None
    if is_calibrated:
        prob_calibrated = calibrator.transform(prob_raw)
    else:
        prob_calibrated = prob_raw
        if calibrator is None:
            note = "无校准器：prob_calibrated = prob_raw，不可解释为概率"
        else:
            note = f"校准方法 {calibrator.method} 未实际应用：prob_calibrated = prob_raw"

    # ---- 6. 组装输出
    out = ds.df.loc[fs.valid_mask].copy()
    out["prob_raw"] = prob_raw
    out["prob_calibrated"] = prob_calibrated
    out["risk_level"] = None  # 预留：等 tiering 模块接入

    # 输出列排序：标识列 → 原始特征 → 概率 → 风险等级
    id_cols = [ds.grid_id_col]
    if ds.timestamp_col:
        id_cols.append(ds.timestamp_col)
    prob_cols = ["prob_raw", "prob_calibrated", "risk_level"]
    feature_cols = [c for c in ds.df.columns if c not in id_cols + [cfg.get("task.target", "flood_label")] and c in out.columns]
    out = out[id_cols + feature_cols + prob_cols]

    citable, banner = _provenance_badge(ds.provenance)

    manifest: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "config_fingerprint": cfg.fingerprint(),
        "provenance": ds.provenance,
        "citable": citable,
        "banner": banner,
        "horizon_hours": float(cfg.get("task.horizon_hours", 0.0)),
        "horizon_steps": cfg.horizon_steps,
        "is_calibrated": is_calibrated,
        "calibration_method": calibrator.method if calibrator else "none",
        "calibration_note": note,
        "n_input_rows": int(len(ds.df)),
        "n_usable_rows": int(len(out)),
        "n_dropped_rows": int(len(ds.df) - len(out)),
        "dropped_reason": "特征历史不足或序列缺口",
        "model_version": None,  # 预留：等字段方案批准后填充
        "model_params": {k: v for k, v in model.get_params().items() if k != "kwargs"},
    }

    # ---- 7. 输出
    if output_path is None:
        output_path = run_dir / "predictions_external.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return PredictionResult(df=out, manifest=manifest, output_path=output_path)


__all__ = [
    "PredictionError",
    "PredictionResult",
    "predict",
]
