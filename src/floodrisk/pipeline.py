"""端到端训练流水线（批 2）。

顺序是有讲究的，每一步都在防止"用未来的信息预测过去"：

    load → features → 复核 → 切分 → 剔除特征不足的行 → 调参(仅训练折)
         → 训练(早停看验证集) → 校准(验证集拟合) → 评价(验证集 + 测试集一次)

真值使用纪律：
- 测试集只在最后由 ``predict`` 与 ``evaluate`` 各碰一次，不参与调参、不参与早停、
  不参与校准器拟合、不参与阈值选择；
- 所有产出目录里都带 ``manifest.json``，写清配置指纹、数据来源、每一处样本损失，
  方便答辩现场逐条追溯。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .calibrate import calibration_report, fit_calibrator
from .config import Config
from .evaluate import binary_metrics, curve_data, format_metrics_table, threshold_sweep
from .features import audit_built_features, build_features
from .io import PROVENANCE_SYNTHETIC, Dataset, load_dataset
from .leakage import audit_features
from .splits import cv_splits, make_split_plan
from .train import (
    constant_baseline,
    importance_table,
    permutation_importance_table,
    predict_proba,
    train_model,
)


def _provenance_badge(provenance: str) -> tuple[bool, str]:
    """返回 (citable, banner)。仅显式声明 REAL 才允许引用。"""
    if provenance == "SYNTHETIC":
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


def _assert_label_horizon_verified(ds: Dataset, cfg: Config) -> None:
    """标签窗口口径闸门。

    合成数据的标签由生成器按同一窗口构造，可以直接训练。
    非合成数据必须由数据提供方**书面确认** ``flood_label`` 严格对应
    ``(t, t + horizon_hours]``，否则中止 —— 不是警告后继续。
    """
    if ds.provenance == PROVENANCE_SYNTHETIC:
        return
    if bool(cfg.get("task.label_horizon_verified", False)):
        return
    horizon = float(cfg.get("task.horizon_hours", 0.0))
    raise RuntimeError(
        f"标签窗口口径未验证：数据来源={ds.provenance}，"
        f"task.horizon_hours={horizon}。\n"
        f"必须在数据侧确认 flood_label 严格对应 (t, t+{horizon}h] 后，"
        "设置 task.label_horizon_verified: true 才允许训练与发布预测。\n"
        "注意：改配置不会让标签变成该口径，只是声明它已经是该口径。"
    )


def _new_run_dir(cfg: Config, output_dir: str | Path | None = None) -> Path:
    root = Path(output_dir or cfg.get("output.dir", "outputs"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{cfg.fingerprint()}"
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _diagnostics(
    *,
    metrics: dict[str, Any],
    imbalance: dict[str, Any],
    calibrator,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tuning: dict[str, Any] | None,
    calibration_test: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """把"看着像能交差、其实有坑"的地方主动写进 manifest。

    这些检查不改变结果，只保证不会被自己忽略、也不会在答辩时被打个措手不及。
    """
    out: list[dict[str, str]] = []

    if calibration_test:
        gap = float(calibration_test["mean_predicted"] - calibration_test["observed_rate"])
        if abs(gap) > 0.05:
            out.append(
                {
                    "level": "medium",
                    "code": "calibration_drift",
                    "message": (
                        f"校准后测试集平均预测概率 {calibration_test['mean_predicted']:.3f} 与实际发生率 "
                        f"{calibration_test['observed_rate']:.3f} 相差 {gap:+.3f}。"
                        f"校准器是在验证集上拟合的，若测试段基准率漂移（本数据正是如此），"
                        f"概率水平会整体偏低/偏高。这种情况**三级分级应优先用分位数策略**，"
                        f"而不是绝对概率阈值"
                    ),
                }
            )

    if imbalance.get("scale_pos_weight") and calibrator.method == "none":
        out.append(
            {
                "level": "high",
                "code": "uncalibrated_weighting",
                "message": (
                    "使用了 scale_pos_weight 但未做概率校准。此时输出分数被系统性推高，"
                    "**不能解释为发生概率**，也不能用绝对阈值做三级分级。"
                    "要么把 calibration.method 改成 platt/isotonic，要么三级分级只用分位数策略"
                ),
            }
        )

    tr_pr, te_pr = metrics["train"].get("pr_auc"), metrics["test"].get("pr_auc")
    if tr_pr is not None and te_pr is not None and tr_pr - te_pr > 0.15:
        out.append(
            {
                "level": "high",
                "code": "overfitting",
                "message": (
                    f"训练集 PR-AUC {tr_pr:.3f} 显著高于测试集 {te_pr:.3f}（差 {tr_pr - te_pr:.3f}）。"
                    f"要么模型记住了训练期的特定事件，要么特征对未来没有稳定解释力。"
                    f"报告里必须写这一条，不能只贴训练集指标"
                ),
            }
        )

    rates = {"train": float(np.mean(y_train)), "val": float(np.mean(y_val)), "test": float(np.mean(y_test))}
    nonzero = {k: v for k, v in rates.items() if v > 0}
    if len(nonzero) == 3 and max(nonzero.values()) / max(min(nonzero.values()), 1e-9) >= 2.0:
        out.append(
            {
                "level": "medium",
                "code": "temporal_drift",
                "message": (
                    f"三段正例率差异较大（train {rates['train']:.3%} / val {rates['val']:.3%} / "
                    f"test {rates['test']:.3%}）。时间切分下这是常见现象：验证/测试段的天气"
                    f"过程与训练段不同。写报告时应说明 PR-AUC 受此影响，不能只看单一数字"
                ),
            }
        )

    te = metrics["test"]
    if te.get("pr_auc") and te.get("baseline_pr_auc"):
        lift = te["pr_auc"] / te["baseline_pr_auc"]
        if lift < 1.5:
            out.append(
                {
                    "level": "medium",
                    "code": "weak_signal",
                    "message": (
                        f"测试集 PR-AUC {te['pr_auc']:.3f} 仅为随机基线 {te['baseline_pr_auc']:.3f} 的 "
                        f"{lift:.1f} 倍，区分能力偏弱。检查是否缺少关键驱动变量（如上游汇水、"
                        f"排水实时状态），而不是继续调参"
                    ),
                }
            )

    if tuning and tuning.get("note") and "提升 -" in str(tuning.get("note")):
        out.append(
            {
                "level": "low",
                "code": "tuning_no_gain",
                "message": "超参搜索未超过未调参基线：说明性能瓶颈在特征与数据，不在超参",
            }
        )
    return out


def _window_unit_warnings(cfg: Config) -> list[dict[str, Any]]:
    """lags / rolling_windows 的单位是「步」，不是小时。

    ``time_step_hours != 1`` 时，同一组取值对应的实际时间跨度会变，
    必须显式告警，避免更换时间粒度时特征的时间含义被悄悄改写。
    """
    step = float(cfg.get("task.time_step_hours", 1.0))
    if abs(step - 1.0) < 1e-9:
        return []
    lags = [int(k) for k in (cfg.get("features.temporal.lags", []) or [])]
    windows = [int(w) for w in (cfg.get("features.temporal.rolling_windows", []) or [])]
    return [
        {
            "level": "medium",
            "code": "window_units_in_steps",
            "message": (
                f"lags / rolling_windows 以「步」计，当前 task.time_step_hours={step}："
                f"lags={lags} 实际为 {[round(k * step, 3) for k in lags]} 小时；"
                f"windows={windows} 实际为 {[round(w * step, 3) for w in windows]} 小时。"
                "更换时间粒度时必须重新确认这些取值，否则特征的时间含义被静默改变"
            ),
        }
    ]


def run_training(
    cfg: Config,
    *,
    output_dir: str | Path | None = None,
    n_iter: int | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """跑完批 2 全流程，返回 manifest。"""
    say = log or (lambda *_: None)
    run_dir = _new_run_dir(cfg, output_dir)
    say(f"运行目录：{run_dir}")

    # ---- 1. 数据 ----
    ds: Dataset = load_dataset(cfg)
    say(f"读入 {len(ds.df)} 行 / {ds.report['stats']['n_grids']} 网格 · 来源={ds.provenance}")
    _assert_label_horizon_verified(ds, cfg)
    cfg.dump(run_dir / "config_snapshot.yaml")
    ds.write_report(run_dir / "validation_report.json")

    # ---- 2. 特征 ----
    fs = build_features(ds, cfg)
    say(f"特征 {fs.report['n_features']} 列；可用行 {fs.report['usable_rows']}，"
        f"剔除 {fs.report['unusable_rows']}")
    audit = audit_built_features(ds, fs, cfg)
    if not audit["passed"]:
        _dump_json(run_dir / "feature_audit_FAILED.json", audit)
        raise RuntimeError(
            f"特征复核未通过：lag 失败 {len(audit['lag_failures'])} 项、"
            f"rolling 失败 {len(audit['rolling_failures'])} 项、"
            f"预报窗口失败 {len(audit['forecast_failures'])} 项，详见 feature_audit_FAILED.json"
        )

    # ---- 4. 切分 ----
    plan = make_split_plan(ds, cfg)
    # 特征不足的行（历史不够/缺口）从三个集合中统一剔除；切分边界不受影响
    keep = fs.valid_mask
    tr = plan.train_idx[keep[plan.train_idx]]
    va = plan.val_idx[keep[plan.val_idx]]
    te = plan.test_idx[keep[plan.test_idx]]
    # 动态模式每个网格贡献多条样本，静态模式一个网格只有一行 —— 下限不同
    min_needed = 10 if cfg.mode == "static" else 30
    if min(len(tr), len(va), len(te)) < min_needed:
        raise RuntimeError(
            f"剔除特征缺失行后样本不足（各段至少需 {min_needed} 行）："
            f"train={len(tr)} val={len(va)} test={len(te)}。"
            f"动态模式可减少 lags 中的长滞后或扩大时间跨度；静态模式请增加网格数"
        )
    y = ds.y
    say(f"切分：{plan.name}  train={len(tr)} val={len(va)} test={len(te)}  "
        f"（因特征缺失再剔除 {len(plan.train_idx) - len(tr)}/{len(plan.val_idx) - len(va)}/"
        f"{len(plan.test_idx) - len(te)}）")

    # ---- 5. 泄漏审计（单变量 AUC 扫描仅限训练段，防止标签回流） ----
    leak = audit_features(ds, cfg, scan_idx=tr)

    X = fs.X

    # ---- 6. 调参（只用训练段） ----
    tuning_payload: dict[str, Any] | None = None
    best_params: dict[str, Any] = {}
    if cfg.get("tuning.enabled", False):
        from .tuning import random_search

        folds_rel = cv_splits(ds, cfg, tr)
        say(f"超参搜索：{n_iter or cfg.get('tuning.n_iter')} 组 × {len(folds_rel)} 折（折含 purge）")
        tun = random_search(X, y, folds_rel, cfg, n_iter=n_iter, log=say)
        best_params = tun.best_params
        tun.table.to_csv(run_dir / "tuning_trials.csv", index=False)
        tuning_payload = tun.describe()
        say(f"最优参数：{best_params}")

    # ---- 6. 训练 ----
    res = train_model(X.iloc[tr], y[tr], X.iloc[va], y[va], cfg, params=best_params)
    say(f"训练完成：best_iteration={res.best_iteration}  imbalance={res.imbalance['strategy']}")

    # ---- 7. 预测与校准 ----
    p_val_raw = predict_proba(res.model, X.iloc[va], best_iteration=res.best_iteration)
    p_test_raw = predict_proba(res.model, X.iloc[te], best_iteration=res.best_iteration)
    p_train_raw = predict_proba(res.model, X.iloc[tr], best_iteration=res.best_iteration)

    cal = fit_calibrator(
        y[va], p_val_raw,
        method=cfg.get("calibration.method", "none"),
        min_samples=int(cfg.get("calibration.min_val_samples", 200)),
    )
    p_val = cal.transform(p_val_raw)
    p_test = cal.transform(p_test_raw)

    # 阈值：在验证集上取 F1 最优点，冻结后套用到测试集。**不是**分级策略，
    # 只是给运行点一个可解释的参考；三级切点由 tiering 模块单独决定。
    sweep = threshold_sweep(y[va], p_val)
    sweep.to_csv(run_dir / "threshold_sweep_val.csv", index=False)
    ref_thr = float(sweep.loc[sweep["f1"].idxmax(), "threshold"])

    metrics: dict[str, Any] = {}
    for name, idx, p_raw, p_cal in (
        ("train", tr, p_train_raw, cal.transform(p_train_raw)),
        ("val", va, p_val_raw, p_val),
        ("test", te, p_test_raw, p_test),
    ):
        metrics[name] = binary_metrics(y[idx], p_cal, threshold=0.5, label=name)
        metrics[f"{name}_at_ref_threshold"] = binary_metrics(y[idx], p_cal, threshold=ref_thr, label=name)
        metrics[name]["operating_point"] = {
            "threshold": ref_thr,
            "source": "验证集 F1 最优点，冻结后套用",
        }
        metrics[name]["calibration"] = cal.method

    metrics["constant_baseline_test"] = constant_baseline(y[tr], y[te])
    metrics["constant_baseline_test"]["note"] += "（在测试集上，用训练集正例率作常数预测）"

    say("\n" + format_metrics_table(
        {k: metrics[k] for k in ("train", "val", "test")}
    ))
    say(f"测试集 @阈值{ref_thr:.2f}: precision={metrics['test_at_ref_threshold']['precision']:.3f} "
        f"recall={metrics['test_at_ref_threshold']['recall']:.3f} "
        f"f1={metrics['test_at_ref_threshold']['f1']:.3f}")

    # ---- 8. 重要性与校准报告 ----
    imp = importance_table(res.model, fs.names)
    imp.to_csv(run_dir / "feature_importance.csv")
    imp_head = imp.head(10).reset_index().to_dict("records")
    try:
        perm = permutation_importance_table(
            res.model, X.iloc[va], y[va],
            repeats=int(cfg.get("evaluation.permutation_repeats", 5)),
            random_state=int(cfg.get("train.random_state", 42)),
        )
        perm.to_csv(run_dir / "feature_importance_permutation.csv", index=False)
        perm_top = perm.head(10).to_dict("records")
    except Exception as exc:  # pragma: no cover
        perm_top = [{"error": str(exc)[:200]}]

    cal_report = calibration_report(y[te], p_test_raw, cal)
    cal_report["val"] = calibration_report(y[va], p_val_raw, cal)

    # ---- 9. 落盘 ----
    if cfg.get("output.save_model", True):
        res.model.save_model(str(run_dir / "model.json"))
    _dump_json(run_dir / "calibrator.json", _calibrator_payload(cal))

    preds = pd.DataFrame(
        {
            "split": np.concatenate([np.full(len(i), n) for n, i in (("val", va), ("test", te))]),
            "row_index": np.concatenate([va, te]),
            "grid_id": ds.df.loc[np.concatenate([va, te]), ds.grid_id_col].to_numpy(),
            "timestamp": (
                ds.df.loc[np.concatenate([va, te]), "timestamp"].astype(str).to_numpy()
                if "timestamp" in ds.df.columns
                else ""
            ),
            "y_true": y[np.concatenate([va, te])],
            "prob_raw": np.round(np.concatenate([p_val_raw, p_test_raw]), 6),
            "prob_calibrated": np.round(np.concatenate([p_val, p_test]), 6),
        }
    )
    preds.to_csv(run_dir / "predictions_val_test.csv", index=False)

    _dump_json(run_dir / "curves_test.json", curve_data(y[te], p_test))

    warnings = _diagnostics(
        metrics=metrics,
        imbalance=res.imbalance,
        calibrator=cal,
        y_train=y[tr],
        y_val=y[va],
        y_test=y[te],
        tuning=tuning_payload,
        calibration_test=cal_report["after"],
    )
    warnings.extend(_window_unit_warnings(cfg))
    for w in warnings:
        say(f"  [{w['level']:6s}] {w['code']}: {w['message'][:110]}")

    citable, banner = _provenance_badge(ds.provenance)
    manifest = {
        "run_id": run_dir.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": ds.provenance,
        "citable": citable,
        "banner": banner,
        "config_fingerprint": cfg.fingerprint(),
        "task": {
            "mode": cfg.mode,
            "anchor": cfg.anchor,
            "horizon_steps": cfg.horizon_steps,
            "horizon_hours": float(cfg.get("task.horizon_hours", 0.0)),
            "label_horizon_verified": bool(cfg.get("task.label_horizon_verified", False)),
            "min_feature_lag": int(cfg.min_feature_lag),
            "min_lag_policy": cfg.get("features.temporal.min_lag_policy", None),
            "target": ds.target,
        },
        "data": {
            "stats": ds.report["stats"],
            "label": ds.report["label"],
            "features_used": ds.features,
            "forecast_columns": ds.forecast_columns,
            "features_dropped": ds.features_dropped,
            "issues": ds.report["issues"],
        },
        "features": fs.report,
        "feature_audit": audit,
        "leakage_audit": leak,
        "split": plan.summary_with_labels(y),
        "row_selection": {
            "split_sizes": {"train": len(plan.train_idx), "val": len(plan.val_idx), "test": len(plan.test_idx)},
            "after_feature_mask": {"train": len(tr), "val": len(va), "test": len(te)},
        },
        "tuning": tuning_payload,
        "model": res.describe(),
        "calibration": cal_report,
        "metrics": metrics,
        "warnings": warnings,
        "importance_top10": imp_head,
        "permutation_importance_top10": perm_top,
        "artifacts": {
            "config": "config_snapshot.yaml",
            "validation_report": "validation_report.json",
            "model": "model.json",
            "predictions": "predictions_val_test.csv",
            "threshold_sweep": "threshold_sweep_val.csv",
            "importance": "feature_importance.csv",
            "curves": "curves_test.json",
        },
    }
    _dump_json(run_dir / "manifest.json", manifest)
    say(f"manifest 已写出：{run_dir / 'manifest.json'}")
    return manifest


def _calibrator_payload(cal) -> dict[str, Any]:
    payload: dict[str, Any] = cal.describe()
    if cal.method == "isotonic" and cal.model is not None:
        payload["x_thresholds"] = np.asarray(cal.model.X_thresholds_).round(8).tolist()
        payload["y_thresholds"] = np.asarray(cal.model.y_thresholds_).round(8).tolist()
    return payload
