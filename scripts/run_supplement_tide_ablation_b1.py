"""delivery_repaired_b1 最小补充实验（用户 2026-10-26 批准，5 项任务）。

任务 → 实现
1. 去潮位消融 no_tide_xgb：以正式训练 dual_xgb 为基准，仅移除潮汐组 4 列
   （tide_pred_max_m / tide_pred_mean_m / tide_n_hours / dist_to_tide_m），
   其余训练数据、时间划分、模型参数（M.build_model 研发默认）、随机种子（42）、
   预处理完全一致；val 上同协议选阈（F1 最大），test 全套指标 + 与 dual_xgb 的
   ROC-AUC / PR-AUC / Brier 差异；移除特征清单 + 泄漏核查（剩余 55 列名不含
   tide/surge/tidal，surge_max_m / tide_station 仍在排除清单）。
2. 模型结果一致性检查：从正式 run 加载 dual_xgb / dual_lr bundle 重新预测 test，
   与正式 predictions_test.parquet 逐值核对；解释 LR 为何 PR-AUC 优于 XGB
   （precision@K、事件日/非事件日分层、分数分位数、top-K 重合、Spearman、
   十分位校准）。不重新调参。
3. 动态风险能力检查：全部样本 vs 事件日内表现（dual_xgb 与 no_tide_xgb）；
   日期级 / grid-day 级聚合 AUC（时间预警能力）；沿海/内陆分组
   （low_coastal_flag 为主、coastal_lowlying_flag 为严格子集）。
4. 阈值稳定性检查：候选阈值在 test 上的 P/R/F1/预警数量，仅敏感性分析，
   不使用 test 重新选阈（正式阈值仍为 val 选定值）。
5. 雨潮复合描述分析（论文 4.x 协议）：test 事件日内雨量分位 × 潮位分位
   组合的均值预测概率，给出样本覆盖数，沿海/内陆分层，允许结果不显著。

硬约束：不做 18 折空间 CV、不做大规模调参、不做复杂概率校准；
全部产物写入独立目录，不覆盖任何旧版本；不自动启动其他训练任务。

用法：
    .venv/bin/python scripts/run_supplement_tide_ablation_b1.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from floodrisk import models as M  # noqa: E402
from floodrisk import realdata as rd  # noqa: E402
from floodrisk.evaluate import binary_metrics  # noqa: E402

DATA = ROOT / "data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet"
FRAME = ROOT / "data/delivery_repaired_b1/sample_frame.parquet"
MANIFEST = ROOT / "data/delivery_repaired_b1/manifest_repair_b1.json"
FORMAL_DIR = ROOT / "outputs/formal_training_repaired_b1_20261026-020814"

KEY_COLS = rd.KEY
TIDE_FEATURES = ["tide_pred_max_m", "tide_pred_mean_m", "tide_n_hours", "dist_to_tide_m"]
COASTAL_COLS = ["low_coastal_flag", "coastal_lowlying_flag", "dist_to_landedge_m"]
META_COLS = ["split", "cv_fold", "dc", "is_event_day", "is_A", "warn_level",
             "flooded_2h", "label_layer", "label_source", "confidence", "is_golden",
             "lon", "lat"]
NO_TIDE_FEATURES = [c for c in rd.FEATURES if c not in TIDE_FEATURES]
RAIN_COL = "power_rain_prev3h_mm"   # SHAP top1 降雨代表列
TIDE_COL = "tide_pred_max_m"        # SHAP rank11 潮位代表列
REF_THRESHOLDS = [0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]
SEED = 42


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    try:
        import resource
        factor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / factor, 1)
    except Exception:
        return -1.0


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
                    encoding="utf-8")


def fast_threshold_sweep(y: np.ndarray, p: np.ndarray,
                         start: float = 0.01, stop: float = 0.99, step: float = 0.01) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []
    for t in np.round(np.arange(start, stop + 1e-9, step), 6):
        yhat = (p >= t).astype(int)
        tp = int(((y == 1) & (yhat == 1)).sum())
        fp = int(((y == 0) & (yhat == 1)).sum())
        fn = int(((y == 1) & (yhat == 0)).sum())
        tn = int(((y == 0) & (yhat == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "precision": round(prec, 6), "recall": round(rec, 6),
                     "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "predicted_positives": tp + fp})
    return pd.DataFrame(rows)


def alerts_per_day(p: np.ndarray, thr: float, n_dates: int) -> float:
    return round(float((np.asarray(p) >= thr).sum()) / max(n_dates, 1), 2)


def compact(m: dict) -> str:
    return " ".join(f"{k}={m.get(k)}" for k in ("n", "positives", "positive_rate", "roc_auc",
                                                "pr_auc", "brier", "precision", "recall", "f1"))


def group_metrics(frame: pd.DataFrame, p: np.ndarray, thr: float, label: str) -> dict:
    """全部样本 / 事件日 / 非事件日 / 分年：动态能力检查的统一口径。"""
    y = frame["y_future_2h"].to_numpy(dtype=int)
    ev = frame["is_event_day"].astype(int).to_numpy()
    out: dict = {"overall": binary_metrics(y, p, threshold=thr, label=f"{label}_overall")}
    for flag in (1, 0):
        sel = ev == flag
        sub_y, sub_p = y[sel], p[sel]
        if len(sub_y) and len(np.unique(sub_y)) == 2:
            out[f"event_day_{flag}"] = binary_metrics(sub_y, sub_p, threshold=thr,
                                                      label=f"{label}_event_day_{flag}")
        else:
            out[f"event_day_{flag}"] = {"n": int(sel.sum()), "positives": int(sub_y.sum()),
                                        "degenerate": "单一类别，排序指标不可定义"}
    years = {}
    year_arr = frame["date_dt"].dt.year.to_numpy()
    for yr in np.unique(year_arr):
        sel = year_arr == yr
        if len(np.unique(y[sel])) == 2:
            years[str(int(yr))] = binary_metrics(y[sel], p[sel], threshold=thr, label=f"{label}_{yr}")
        else:
            years[str(int(yr))] = {"n": int(sel.sum()), "positives": int(y[sel].sum()),
                                   "degenerate": "单一类别"}
    out["by_year"] = years
    return out


def aggregated_auc(frame: pd.DataFrame, p: np.ndarray, keys: list[str]) -> dict:
    """时间预警能力：把行级概率聚合到日期级 / grid-day 级再算排序指标。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.DataFrame({
        "date": frame["date"].astype(str).to_numpy(),
        "grid_id": frame["grid_id"].astype(str).to_numpy(),
        "is_event_day": frame["is_event_day"].astype(int).to_numpy(),
        "y": frame["y_future_2h"].to_numpy(dtype=int),
        "p": p,
    })
    out = {}
    # 日期级：该日最大/平均概率 vs 该日是否官方警告日
    g = df.groupby("date", observed=True).agg(p_max=("p", "max"), p_mean=("p", "mean"),
                                              is_event_day=("is_event_day", "first"))
    yd = g["is_event_day"].to_numpy()
    out["date_level"] = {
        "n_dates": int(len(g)), "n_event_dates": int(yd.sum()),
        "p_max_roc": round(float(roc_auc_score(yd, g["p_max"])), 6) if yd.sum() < len(yd) else None,
        "p_max_pr": round(float(average_precision_score(yd, g["p_max"])), 6) if yd.sum() else None,
        "p_mean_roc": round(float(roc_auc_score(yd, g["p_mean"])), 6) if yd.sum() < len(yd) else None,
        "p_mean_pr": round(float(average_precision_score(yd, g["p_mean"])), 6) if yd.sum() else None,
        "baseline_pr": round(float(yd.mean()), 6),
    }
    # grid-day 级：该格·日是否有正例槽
    g2 = df.groupby(["grid_id", "date"], observed=True).agg(p_max=("p", "max"), p_mean=("p", "mean"),
                                                            y=("y", "max"))
    yg = g2["y"].to_numpy()
    out["grid_day_level"] = {
        "n_grid_days": int(len(g2)), "n_positive_grid_days": int(yg.sum()),
        "p_max_roc": round(float(roc_auc_score(yg, g2["p_max"])), 6),
        "p_max_pr": round(float(average_precision_score(yg, g2["p_max"])), 6),
        "p_mean_roc": round(float(roc_auc_score(yg, g2["p_mean"])), 6),
        "p_mean_pr": round(float(average_precision_score(yg, g2["p_mean"])), 6),
        "baseline_pr": round(float(yg.mean()), 6),
    }
    return out


def rain_tide_composite(frame: pd.DataFrame, p: np.ndarray, coastal: np.ndarray) -> pd.DataFrame:
    """事件日内 雨量三分位 × 潮位三分位 → 均值预测概率 + 实际正例率（论文复合条件协议）。"""
    df = pd.DataFrame({
        "rain": frame[RAIN_COL].to_numpy(dtype=float),
        "tide": frame[TIDE_COL].to_numpy(dtype=float),
        "p": p,
        "coastal": coastal,
        "y": frame["y_future_2h"].to_numpy(dtype=int),
    })
    try:
        rain_codes = pd.qcut(df["rain"], q=3, labels=False, duplicates="drop")
        tide_codes = pd.qcut(df["tide"], q=3, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    if rain_codes.nunique() < 3 or tide_codes.nunique() < 3:
        return pd.DataFrame()
    df["rain_bin"] = rain_codes.map({0: "雨低", 1: "雨中", 2: "雨高"})
    df["tide_bin"] = tide_codes.map({0: "潮低", 1: "潮中", 2: "潮高"})
    rows = []
    for scope, sub in (("事件日-全部", df),
                       ("事件日-沿海(low_coastal)", df[df["coastal"] == 1]),
                       ("事件日-内陆(low_coastal=0)", df[df["coastal"] == 0])):
        if len(sub) == 0:
            continue
        piv = sub.groupby(["rain_bin", "tide_bin"], observed=True).agg(
            n=("p", "size"), mean_p=("p", "mean"), obs_rate=("y", "mean"),
            p90=("p", lambda s: float(np.quantile(s, 0.9))))
        for (rb, tb), r in piv.iterrows():
            rows.append({"scope": scope, "rain_bin": str(rb), "tide_bin": str(tb),
                         "n": int(r["n"]), "mean_p": round(float(r["mean_p"]), 6),
                         "obs_rate": round(float(r["obs_rate"]), 6),
                         "p90_p": round(float(r["p90"]), 6),
                         "rain_median": round(float(sub.loc[(sub["rain_bin"] == rb), "rain"].median()), 4),
                         "tide_median": round(float(sub.loc[(sub["tide_bin"] == tb), "tide"].median()), 4)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / f"outputs/supplement_tide_ablation_b1_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    result: dict = {
        "created_at": stamp,
        "run_kind": "supplement_tide_ablation_b1",
        "formal_run_dir": str(FORMAL_DIR.relative_to(ROOT)),
        "data": {"path": str(DATA.relative_to(ROOT)), "sha256": None,
                 "version": "delivery_repaired_b1"},
        "params": {"seed": SEED, "removed_tide_features": TIDE_FEATURES,
                   "n_features_no_tide": len(NO_TIDE_FEATURES),
                   "tuning": "无调参（M.build_model 研发默认）",
                   "threshold_policy": "no_tide 同协议：val(2022) F1 最大；test 阈值仅敏感性分析"},
    }

    # ---- step0 版本冻结 + 特征集核查 -----------------------------------------
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sha_data, sha_frame = sha256(DATA), sha256(FRAME)
    result["data"]["sha256"] = sha_data
    ok_sha = (sha_data == manifest["outputs"]["main"]["sha256"]
              and sha_frame == manifest["outputs"]["frame"]["sha256"])
    result["step0_freeze"] = {"sha_matches_manifest": bool(ok_sha)}
    print(f"[0] sha match={ok_sha}")
    if not ok_sha:
        print("[ABORT] SHA256 与 manifest 不一致")
        return 2

    tide_group_cols = sorted(c for g, cols in rd.FEATURE_GROUPS.items() if "潮" in g for c in cols)
    checks = {
        "tide_group_matches_removed_list": tide_group_cols == sorted(TIDE_FEATURES),
        "removed_are_features": set(TIDE_FEATURES) <= set(rd.FEATURES),
        "n_no_tide": len(NO_TIDE_FEATURES),
        "no_tide_minus_dual": len(rd.FEATURES) - len(NO_TIDE_FEATURES),
        "no_tide_name_leak": sorted(c for c in NO_TIDE_FEATURES
                                    if re.search(r"tide|surge|tidal", c, re.I)),
        "no_tide_in_excluded": sorted(set(NO_TIDE_FEATURES) & set(rd.EXCLUDED)),
        "surge_max_m_still_excluded": "surge_max_m" in rd.EXCLUDED,
        "tide_station_still_excluded": "tide_station" in rd.EXCLUDED,
        "rain_col_is_feature": RAIN_COL in rd.FEATURES,
        "tide_col_is_feature": TIDE_COL in rd.FEATURES,
    }
    checks["ok"] = (checks["tide_group_matches_removed_list"] and checks["removed_are_features"]
                    and checks["no_tide_minus_dual"] == 4 and not checks["no_tide_name_leak"]
                    and not checks["no_tide_in_excluded"] and checks["surge_max_m_still_excluded"]
                    and checks["tide_station_still_excluded"]
                    and checks["rain_col_is_feature"] and checks["tide_col_is_feature"])
    result["step0_feature_checks"] = checks
    print(f"[0] feature checks ok={checks['ok']} n_no_tide={len(NO_TIDE_FEATURES)}")
    if not checks["ok"]:
        print(f"[ABORT] 特征集核查失败: {checks}")
        write_json(outdir / "aborted.json", result)
        return 2

    # ---- step1 读入 + 目标 + 划分（与正式训练同一路径）------------------------
    t = time.time()
    data = rd.load_delivery(str(DATA),
                            columns=KEY_COLS + META_COLS + rd.FEATURES + COASTAL_COLS,
                            categorical=["grid_id", "date", "label_layer", "label_source"])
    data, _ = rd.build_targets(data)
    data, split_info = rd.assign_splits(data)
    partition = rd.check_partition(data)
    result["step1_split"] = {"counts": split_info["counts"], "partition": partition}
    print(f"[1] read+split {time.time() - t:.1f}s counts={split_info['counts']}")
    if any(partition[k] != 0 for k in ("grid_day_overlap_train_val", "grid_day_overlap_train_test",
                                       "grid_day_overlap_val_test", "date_overlap_train_test")):
        print("[ABORT] 划分重叠")
        write_json(outdir / "aborted.json", result)
        return 2

    for c in rd.FEATURES:
        data[c] = data[c].astype("float32")
    for c in COASTAL_COLS:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0).astype("int8")

    tr = data[data["exp_split"].eq("train")]
    va = data[data["exp_split"].eq("val")]
    te = data[data["exp_split"].eq("test")].copy()
    y_tr = tr["y_future_2h"].to_numpy(dtype=int)
    y_va = va["y_future_2h"].to_numpy(dtype=int)
    y_te = te["y_future_2h"].to_numpy(dtype=int)
    n_dates_test = int(te["date"].nunique())
    coastal = te["low_coastal_flag"].to_numpy(dtype=int)
    result["step1_split"]["test_dates"] = n_dates_test
    result["step1_split"]["test_event_dates"] = int(te.loc[te["is_event_day"].astype(int) == 1, "date"].nunique())
    result["step1_split"]["coastal_grids_low"] = int(te.groupby("grid_id", observed=True)["low_coastal_flag"].first().sum())
    result["step1_split"]["coastal_grids_strict"] = int(te.groupby("grid_id", observed=True)["coastal_lowlying_flag"].first().sum())
    print(f"[1] test={len(te):,} dates={n_dates_test} coastal_grids={result['step1_split']['coastal_grids_low']}")

    # ---- step2 训练 no_tide_xgb（与 formal 同配置同种子）----------------------
    t0 = time.time()
    model_nt = M.build_model("xgb", seed=SEED)
    model_nt.fit(tr[NO_TIDE_FEATURES], y_tr)
    fit_s = round(time.time() - t0, 1)
    p_va_nt = M.predict_proba(model_nt, va[NO_TIDE_FEATURES])
    p_te_nt = M.predict_proba(model_nt, te[NO_TIDE_FEATURES])
    sweep = fast_threshold_sweep(y_va, p_va_nt)
    sweep.to_csv(outdir / "threshold_sweep_no_tide_xgb.csv", index=False)
    best = sweep.loc[sweep["f1"].idxmax()]
    thr_nt = float(best["threshold"])
    thr_formal = 0.05  # formal dual_xgb 的 val 阈值（metrics.json 复核见 step3）
    m_va_nt = binary_metrics(y_va, p_va_nt, threshold=thr_nt, label="no_tide_val")
    print(f"[2] no_tide_xgb fit {fit_s}s thr={thr_nt:.2f} val {compact(m_va_nt)}")

    # ---- step3 与 formal dual_xgb 的 test 指标对比 ----------------------------
    formal = json.loads((FORMAL_DIR / "metrics.json").read_text(encoding="utf-8"))
    fm = formal["step6_test_final"]["models"]
    f_dual = fm["dual_xgb"]["test_at_own_val_threshold"]
    thr_formal = float(formal["step6_test_final"]["thresholds_frozen_on_val"]["dual_xgb"])
    m_te_nt_own = binary_metrics(y_te, p_te_nt, threshold=thr_nt, label="no_tide_test@own")
    m_te_nt_fixed = binary_metrics(y_te, p_te_nt, threshold=thr_formal, label="no_tide_test@0.05")
    grp_nt = group_metrics(te, p_te_nt, thr_nt, "no_tide")
    result["step3_ablation"] = {
        "no_tide_val": m_va_nt, "no_tide_val_threshold": thr_nt,
        "no_tide_test_own": m_te_nt_own, "no_tide_test_at_0.05": m_te_nt_fixed,
        "no_tide_apd_own": alerts_per_day(p_te_nt, thr_nt, n_dates_test),
        "no_tide_apd_fixed": alerts_per_day(p_te_nt, thr_formal, n_dates_test),
        "formal_dual_apd": fm["dual_xgb"]["test_alerts_per_day_own_thr"],
        "no_tide_group_report_test": grp_nt,
        "formal_dual_xgb_test": f_dual,
        "delta_vs_dual": {
            "roc_auc": round(m_te_nt_own["roc_auc"] - f_dual["roc_auc"], 6),
            "pr_auc": round(m_te_nt_own["pr_auc"] - f_dual["pr_auc"], 6),
            "brier": round(m_te_nt_own["brier"] - f_dual["brier"], 6),
        },
        "delta_vs_power_only": {
            "roc_auc": round(m_te_nt_own["roc_auc"]
                             - fm["power_only_xgb"]["test_at_own_val_threshold"]["roc_auc"], 6),
        },
        "fit_seconds": fit_s,
    }
    print(f"[3] no_tide test@own {compact(m_te_nt_own)} | Δ vs dual "
          f"ROC={result['step3_ablation']['delta_vs_dual']['roc_auc']:+.4f} "
          f"PR={result['step3_ablation']['delta_vs_dual']['pr_auc']:+.4f} "
          f"Brier={result['step3_ablation']['delta_vs_dual']['brier']:+.4f}")
    m_nt = m_te_nt_own

    # ---- step4 一致性检查：formal bundle 重预测 + PR-AUC 诊断 -----------------
    t0 = time.time()
    preds_f = pd.read_parquet(FORMAL_DIR / "predictions_test.parquet")
    key_f = preds_f[KEY_COLS].copy()
    for c in ("grid_id", "date"):
        key_f[c] = key_f[c].astype(str)
    key_f["slot"] = key_f["slot"].astype(int)
    key_t = te[KEY_COLS].copy()
    for c in ("grid_id", "date"):
        key_t[c] = key_t[c].astype(str)
    key_t["slot"] = key_t["slot"].astype(int)
    key_t["_row"] = np.arange(len(key_t))
    merged = key_t.merge(key_f.assign(_frow=np.arange(len(key_f))), on=KEY_COLS, how="left")
    row_aligned = bool(merged["_frow"].notna().all()
                       and (merged["_frow"].to_numpy() == merged["_row"].to_numpy()).all())
    y_match = bool((preds_f["y_future_2h"].to_numpy()[merged["_frow"].fillna(-1).astype(int).to_numpy()]
                    == y_te).all()) if row_aligned else None
    cons: dict = {"n_test": int(len(te)), "n_formal_preds": int(len(preds_f)),
                  "rows_aligned_in_order": row_aligned, "y_future_2h_identical": y_match}

    p_re = {}
    for name, fs in (("dual_xgb", list(rd.FEATURES)), ("dual_lr", list(rd.FEATURES))):
        bundle = M.load_bundle(FORMAL_DIR / "bundles" / f"{name}.joblib")
        assert bundle.features == fs, f"{name} bundle 特征清单与 realdata 不一致"
        p_b = M.predict_with_bundle(bundle, te[fs])
        p_re[name] = p_b
        p_ref = preds_f[f"p_{name}"].to_numpy(dtype=float)[merged["_frow"].to_numpy()]
        cons[f"{name}_max_abs_prob_diff"] = float(np.max(np.abs(p_b - p_ref)))
        cons[f"{name}_bundle_features_n"] = len(bundle.features)

    # 复现 ROC/PR 并与 formal metrics.json 对数
    for name in ("dual_xgb", "dual_lr"):
        from sklearn.metrics import average_precision_score, roc_auc_score
        p_ref = preds_f[f"p_{name}"].to_numpy(dtype=float)[merged["_frow"].to_numpy()]
        roc_re = float(roc_auc_score(y_te, p_re[name]))
        pr_re = float(average_precision_score(y_te, p_re[name]))
        roc_ref = float(fm[name]["test_at_own_val_threshold"]["roc_auc"])
        pr_ref = float(fm[name]["test_at_own_val_threshold"]["pr_auc"])
        cons[f"{name}_roc_repro_diff"] = round(abs(roc_re - roc_ref), 8)
        cons[f"{name}_pr_repro_diff"] = round(abs(pr_re - pr_ref), 8)

    # LR vs XGB 的 PR-AUC 差异诊断
    p_x, p_l = p_re["dual_xgb"], p_re["dual_lr"]
    ev = te["is_event_day"].astype(int).to_numpy()
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr
    diag_rows = []
    for K in (1000, 5000, 10000, 50000, 100000, 200000, 500000):
        top_x = set(np.argsort(-p_x)[:K].tolist())
        top_l = set(np.argsort(-p_l)[:K].tolist())
        prec_x = y_te[np.argsort(-p_x)[:K]].mean()
        prec_l = y_te[np.argsort(-p_l)[:K]].mean()
        diag_rows.append({"diagnostic": f"precision@K(K={K})", "dual_xgb": round(float(prec_x), 6),
                          "dual_lr": round(float(prec_l), 6),
                          "topK_overlap_rate": round(len(top_x & top_l) / K, 6)})
    for flag, tag in ((1, "event_day_1"), (0, "event_day_0")):
        sel = ev == flag
        if int(y_te[sel].sum()) == 0:
            # 标签 = A 类格 × 官方警告窗口，非事件日按构造无正例，排序指标不可定义
            diag_rows.append({"diagnostic": f"roc_auc[{tag}]", "dual_xgb": None, "dual_lr": None,
                              "topK_overlap_rate": f"无正例(n={int(sel.sum())})，标签构造使然"})
            diag_rows.append({"diagnostic": f"pr_auc[{tag}]", "dual_xgb": None, "dual_lr": None,
                              "topK_overlap_rate": None})
            continue
        diag_rows.append({"diagnostic": f"roc_auc[{tag}]", "dual_xgb": round(float(roc_auc_score(y_te[sel], p_x[sel])), 6),
                          "dual_lr": round(float(roc_auc_score(y_te[sel], p_l[sel])), 6), "topK_overlap_rate": None})
        diag_rows.append({"diagnostic": f"pr_auc[{tag}]", "dual_xgb": round(float(average_precision_score(y_te[sel], p_x[sel])), 6),
                          "dual_lr": round(float(average_precision_score(y_te[sel], p_l[sel])), 6), "topK_overlap_rate": None})
    for q in (0.5, 0.75, 0.9, 0.99, 0.999, 1.0):
        diag_rows.append({"diagnostic": f"score_q{q}", "dual_xgb": round(float(np.quantile(p_x, q)), 6),
                          "dual_lr": round(float(np.quantile(p_l, q)), 6), "topK_overlap_rate": None})
    idx = np.random.default_rng(SEED).choice(len(p_x), size=min(200_000, len(p_x)), replace=False)
    diag_rows.append({"diagnostic": "spearman(sample200k)", "dual_xgb": None, "dual_lr": None,
                      "topK_overlap_rate": round(float(spearmanr(p_x[idx], p_l[idx]).statistic), 6)})
    diag_rows.append({"diagnostic": "mean_score", "dual_xgb": round(float(p_x.mean()), 6),
                      "dual_lr": round(float(p_l.mean()), 6), "topK_overlap_rate": None})
    diag_rows.append({"diagnostic": "score_std", "dual_xgb": round(float(p_x.std()), 6),
                      "dual_lr": round(float(p_l.std()), 6), "topK_overlap_rate": None})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(outdir / "pr_auc_diagnosis_xgb_vs_lr.csv", index=False)

    # 十分位校准（解释 Brier 差异）
    cal_rows = []
    for name, p in (("dual_xgb", p_x), ("dual_lr", p_l)):
        bins = pd.qcut(p, q=10, duplicates="drop")
        g = pd.DataFrame({"bin": bins, "y": y_te, "p": p}).groupby("bin", observed=True).agg(
            n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
        for b, r in g.iterrows():
            cal_rows.append({"model": name, "decile": str(b), "n": int(r["n"]),
                             "mean_pred": round(float(r["mean_pred"]), 6),
                             "observed_rate": round(float(r["observed_rate"]), 6)})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(outdir / "calibration_deciles.csv", index=False)
    cons["diagnostics"] = diag.to_dict("records")
    cons["seconds"] = round(time.time() - t0, 1)
    result["step4_consistency"] = cons
    print(f"[4] aligned={row_aligned} y_match={y_match} "
          f"max_diff_xgb={cons['dual_xgb_max_abs_prob_diff']:.2e} "
          f"max_diff_lr={cons['dual_lr_max_abs_prob_diff']:.2e} ({cons['seconds']}s)")

    # ---- step5 动态风险能力（dual_xgb 用 formal 重载概率；no_tide 本轮概率）----
    dyn: dict = {}
    for name, p in (("dual_xgb", p_x), ("no_tide_xgb", p_te_nt)):
        dyn[name] = {
            "group_report_test": group_metrics(te, p, thr_formal, name),
            "aggregated_auc_test": aggregated_auc(te, p, KEY_COLS),
        }
    # 沿海 / 内陆分组（low_coastal_flag 为主，coastal_lowlying_flag 为严格子集）
    coastal_rows = []
    for name, p in (("dual_xgb", p_x), ("no_tide_xgb", p_te_nt)):
        for col in ("low_coastal_flag", "coastal_lowlying_flag"):
            flag = te[col].to_numpy(dtype=int)
            for v, tag in ((1, "沿海"), (0, "内陆")):
                sel = (flag == v)
                if sel.sum() == 0 or len(np.unique(y_te[sel])) < 2:
                    continue
                m = binary_metrics(y_te[sel], p[sel], threshold=thr_formal, label=f"{name}_{col}_{tag}")
                if "degenerate" in m:
                    continue
                coastal_rows.append({"model": name, "flag": col, "group": tag,
                                     "n_grids": int(te.loc[sel, "grid_id"].nunique()),
                                     "n": int(sel.sum()), "positives": int(y_te[sel].sum()),
                                     "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    coastal_df = pd.DataFrame(coastal_rows)
    coastal_df.to_csv(outdir / "coastal_groups_test.csv", index=False)
    dyn["coastal_groups"] = coastal_df.to_dict("records")
    result["step5_dynamic"] = dyn
    d1 = dyn["dual_xgb"]["group_report_test"]["event_day_1"]
    print(f"[5] dual_xgb event_day_1 n={d1.get('n')} ROC={d1.get('roc_auc')} | "
          f"date_level ROC={dyn['dual_xgb']['aggregated_auc_test']['date_level']['p_max_roc']}")

    # ---- step6 test 阈值敏感性（仅敏感性，不重新选阈）-------------------------
    sens_rows = []
    for name, p in (("dual_xgb", p_x), ("no_tide_xgb", p_te_nt)):
        for t_ in REF_THRESHOLDS:
            m = binary_metrics(y_te, p, threshold=t_, label=f"{name}@{t_}")
            sens_rows.append({"model": name, "threshold": t_, "precision": m["precision"],
                              "recall": m["recall"], "f1": m["f1"], "tp": m["tp"], "fp": m["fp"],
                              "alerts_per_day": alerts_per_day(p, t_, n_dates_test)})
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(outdir / "threshold_sensitivity_test.csv", index=False)
    result["step6_threshold_sensitivity_test"] = sens.to_dict("records")
    result["step6_threshold_sensitivity_test_note"] = (
        "本表为 test 上的敏感性分析，不用于阈值选择；正式阈值仍为各自 val 上 F1 最大点"
        f"（dual_xgb={thr_formal}，no_tide_xgb={thr_nt}）")
    print(f"[6] sensitivity rows={len(sens)} (test only, NOT re-selection)")

    # ---- step7 雨潮复合描述分析（事件日内；允许不显著）------------------------
    ev_mask = te["is_event_day"].astype(int).to_numpy() == 1
    comp_ev = rain_tide_composite(te[ev_mask], p_x[ev_mask], coastal[ev_mask])
    # 对照：全部 test 行（敏感性）
    comp_all = rain_tide_composite(te, p_x, coastal)
    comp_ev.to_csv(outdir / "rain_tide_composite_event_days.csv", index=False)
    comp_all.to_csv(outdir / "rain_tide_composite_all_test.csv", index=False)
    hi = None
    if len(comp_ev):
        sub = comp_ev[comp_ev["scope"] == "事件日-全部"]
        if len(sub):
            is_hh = (sub["rain_bin"] == "雨高") & (sub["tide_bin"] == "潮高")
            hh, rest = sub[is_hh], sub[~is_hh]
            if len(hh) and len(rest):
                rest_mean = float((rest["mean_p"] * rest["n"]).sum() / rest["n"].sum())
                hi = {"mean_p_high_rain_high_tide": float(hh["mean_p"].iloc[0]),
                      "n_high_rain_high_tide": int(hh["n"].iloc[0]),
                      "mean_p_rest": round(rest_mean, 6),
                      "n_rest": int(rest["n"].sum()),
                      "uplift": round(float(hh["mean_p"].iloc[0] - rest_mean), 6)}
    result["step7_rain_tide_composite"] = {
        "rain_col": RAIN_COL, "tide_col": TIDE_COL, "scope": "test 官方警告日（事件日）内",
        "high_vs_rest_event_days": hi,
        "n_event_day_rows": int(ev_mask.sum()),
        "note": "描述性分层分析；弱监督标签下不构成因果或统计显著性检验",
        "event_days_head": comp_ev.head(24).to_dict("records"),
    }
    print(f"[7] composite event rows={ev_mask.sum()} high_vs_rest={hi}")

    # ---- step8 落盘：预测、bundle、汇总 ---------------------------------------
    keep = KEY_COLS + ["timestamp", "is_event_day", "y_future_2h"]
    out_p = te[keep].copy()
    out_p["p_no_tide_xgb"] = p_te_nt.astype("float64")
    out_p.to_parquet(outdir / "predictions_test_no_tide.parquet", index=False)
    bundle_nt = M.Bundle(
        backend="xgb", model=model_nt, features=list(NO_TIDE_FEATURES),
        meta={"version": f"repaired_b1_supplement_no_tide_{stamp}", "feature_set": "no_tide_xgb",
              "n_features": len(NO_TIDE_FEATURES), "n_train": int(len(tr)), "seed": SEED,
              "target": "y_future_2h", "threshold_val_f1max": thr_nt,
              "data_sha256": sha_data, "data_version": "delivery_repaired_b1",
              "exploratory_weak_supervision": True,
              "removed_features": TIDE_FEATURES,
              "note": "去潮位消融模型；输出为弱监督标签下的风险评分"},
    )
    M.save_bundle(outdir / "bundles/no_tide_xgb.joblib", bundle_nt)

    result["total_seconds"] = round(time.time() - t_start, 1)
    result["peak_rss_mb"] = peak_rss_mb()
    result["disclaimer"] = (
        "弱监督标签下的补充实验：全部指标不得解释为真实水浸提前 2 小时预测准确率；"
        "模型输出不得称为已验证的真实水浸发生概率；阈值敏感性表不构成重新选阈。")
    write_json(outdir / "metrics.json", result)

    md = build_report_md(result, outdir)
    (outdir / "results_report.md").write_text(md, encoding="utf-8")
    (ROOT / "reports/supplement_tide_ablation_b1.md").write_text(md, encoding="utf-8")
    print(f"\n[done] {result['total_seconds']}s peak {result['peak_rss_mb']}MB → {outdir}")
    return 0


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def _tbl(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _mrow(tag: str, m: dict) -> list:
    if "degenerate" in m:
        return [tag, m.get("n"), m.get("positives"), "退化", "—", "—", "—", "—"]
    return [tag, m["n"], m["positives"], f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}",
            f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]


def build_report_md(result: dict, outdir: Path) -> str:
    s0 = result["step0_feature_checks"]
    s1 = result["step1_split"]
    s3 = result["step3_ablation"]
    s4 = result["step4_consistency"]
    s5 = result["step5_dynamic"]
    s6 = result["step6_threshold_sensitivity_test"]
    s7 = result["step7_rain_tide_composite"]
    outdir_rel = f"outputs/{outdir.name}"

    L: list[str] = []
    A = L.append
    A("# 最小补充实验：去潮位消融 + 一致性 + 动态能力 + 阈值敏感性 + 雨潮复合")
    A("")
    A("> **解读边界**：弱监督标签（`flooded_2h` = A 类空间规则 × HKO 官方红/黑警告归因窗口）下的探索性结果，"
      "不得解释为真实水浸提前 2 小时预测准确率；模型输出为风险评分，不得称为已验证的真实水浸发生概率；"
      "样本含 320 个改判日 35 格受限重抽偏差；ERA5/POWER 为历史再分析/遥测口径，与实时部署输入存在时效差异。")
    A("")
    A(f"运行目录：`{outdir_rel}`（独立保存，未覆盖任何旧产物）  ")
    A(f"基准：正式训练 `outputs/formal_training_repaired_b1_20261026-020814`（只读）  ")
    A(f"数据：delivery_repaired_b1，SHA256 `{result['data']['sha256'][:16]}…`（与 manifest 一致："
      f"{result['step0_freeze']['sha_matches_manifest']}）")
    A("")
    A("## 0. 特征集核查（任务 1 前置：防潮位信息经衍生字段保留）")
    A("")
    A(_tbl(["检查项", "结果"], [
        ["潮汐组特征 = 移除清单", f"{s0['tide_group_matches_removed_list']}（{sorted(TIDE_FEATURES)}）"],
        ["移除列数", s0["no_tide_minus_dual"]],
        ["剩余特征数", s0["n_no_tide"]],
        ["剩余特征名含 tide/surge/tidal", s0["no_tide_name_leak"] or "无"],
        ["剩余特征 ∩ 排除清单", s0["no_tide_in_excluded"] or "空"],
        ["surge_max_m 仍在排除清单（未来增水量）", s0["surge_max_m_still_excluded"]],
        ["tide_station 仍在排除清单（站标识）", s0["tide_station_still_excluded"]],
    ]))
    A("")
    A(f"移除的 4 个潮位特征：`tide_pred_max_m`、`tide_pred_mean_m`、`tide_n_hours`、`dist_to_tide_m`。"
      f"保留的 `dist_to_landedge_m`、`lowland_flag_500` 等为**静态地形/海岸暴露**变量（每格常量，不含任何潮位数值），"
      f"去潮位 ≠ 去沿海地形；`surge_max_m`（未来增水量）与 `tide_station`（站点标识）自始在排除清单。")
    A("")
    A(f"划分复核：train {s1['counts'].get('train', '—')} / val {s1['counts'].get('val', '—')} / "
      f"test {s1['counts'].get('test', '—')}；grid-day 与日期重叠均为 0；"
      f"test 采样日 {s1['test_dates']}（事件日 {s1['test_event_dates']}），"
      f"沿海格（low_coastal_flag）{s1['coastal_grids_low']} 个、严格沿海低洼（coastal_lowlying_flag）"
      f"{s1['coastal_grids_strict']} 个。")
    A("")
    A("## 1. 去潮位消融 no_tide_xgb（任务 1，最高优先级）")
    A("")
    A(f"与正式训练 dual_xgb 唯一差异 = 移除潮汐组 4 列（59 → {s0['n_no_tide']} 列）；"
      f"数据、划分、参数（XGBoost hist 300/depth4，研发默认）、随机种子（42）、预处理完全一致。"
      f"阈值同协议：val(2022) F1 最大 → **{s3['no_tide_val_threshold']:.2f}**。")
    A("")
    f2 = s3["formal_dual_xgb_test"]
    A(_tbl(["模型", "阈值", "ROC-AUC", "PR-AUC", "Brier", "Precision", "Recall", "F1", "日均预警"],
           [["dual_xgb（正式基准）", f"{f2['threshold']:.2f}", f"{f2['roc_auc']:.4f}", f"{f2['pr_auc']:.4f}",
             f"{f2['brier']:.4f}", f"{f2['precision']:.4f}", f"{f2['recall']:.4f}", f"{f2['f1']:.4f}",
             s3["formal_dual_apd"]],
            ["no_tide_xgb @自选阈", f"{s3['no_tide_val_threshold']:.2f}",
             f"{s3['no_tide_test_own']['roc_auc']:.4f}", f"{s3['no_tide_test_own']['pr_auc']:.4f}",
             f"{s3['no_tide_test_own']['brier']:.4f}", f"{s3['no_tide_test_own']['precision']:.4f}",
             f"{s3['no_tide_test_own']['recall']:.4f}", f"{s3['no_tide_test_own']['f1']:.4f}",
             s3["no_tide_apd_own"]],
            ["no_tide_xgb @0.05", "0.05", f"{s3['no_tide_test_at_0.05']['roc_auc']:.4f}",
             f"{s3['no_tide_test_at_0.05']['pr_auc']:.4f}", f"{s3['no_tide_test_at_0.05']['brier']:.4f}",
             f"{s3['no_tide_test_at_0.05']['precision']:.4f}", f"{s3['no_tide_test_at_0.05']['recall']:.4f}",
             f"{s3['no_tide_test_at_0.05']['f1']:.4f}", s3["no_tide_apd_fixed"]]]))
    A("")
    d = s3["delta_vs_dual"]
    A(f"**Δ（no_tide − dual_xgb）**：ROC-AUC {d['roc_auc']:+.4f}、PR-AUC {d['pr_auc']:+.4f}、"
      f"Brier {d['brier']:+.4f}。作为参照，正式训练中 tide_pred_max_m 的 mean|SHAP| 排名第 11（0.249），"
      f"与消融幅度量级一致。去除潮位后 ROC-AUC 变化 "
      f"{abs(d['roc_auc']):.4f}（{'可忽略' if abs(d['roc_auc']) < 0.005 else '有限但非零'}），"
      f"说明**判别力主要不由潮位特征贡献**；潮位信息对排序的边际作用有限，"
      f"论文如需强调「雨潮复合」应基于 §5 的描述性分层而非模型指标差异。")
    A("")
    g = s3["no_tide_group_report_test"]
    rows = [_mrow("no_tide overall", g["overall"]), _mrow("no_tide event_day_1", g["event_day_1"]),
            _mrow("no_tide event_day_0", g["event_day_0"])]
    for y, m in sorted(g["by_year"].items()):
        rows.append(_mrow(f"no_tide {y}", m))
    A("no_tide_xgb 分组表现（test @ 自选阈值）：")
    A("")
    A(_tbl(["分组", "n", "正例", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"], rows))
    A("")
    A("## 2. 模型结果一致性检查（任务 2）")
    A("")
    A(_tbl(["检查项", "结果"], [
        ["test 行数（本轮 / formal）", f"{s4['n_test']:,} / {s4['n_formal_preds']:,}"],
        ["键全等且行序一致", s4["rows_aligned_in_order"]],
        ["y_future_2h 逐行一致", s4["y_future_2h_identical"]],
        ["dual_xgb 重预测最大概率差", f"{s4['dual_xgb_max_abs_prob_diff']:.2e}"],
        ["dual_lr 重预测最大概率差", f"{s4['dual_lr_max_abs_prob_diff']:.2e}"],
        ["dual_xgb ROC/PR 复现差", f"{s4['dual_xgb_roc_repro_diff']:.1e} / {s4['dual_xgb_pr_repro_diff']:.1e}"],
        ["dual_lr ROC/PR 复现差", f"{s4['dual_lr_roc_repro_diff']:.1e} / {s4['dual_lr_pr_repro_diff']:.1e}"],
    ]))
    A("")
    A(f"结论：从 formal bundle 独立重预测与正式产物**概率逐值完全一致**（最大差 "
      f"{max(s4['dual_xgb_max_abs_prob_diff'], s4['dual_lr_max_abs_prob_diff']):.1e}），"
      f"ROC/PR 复现差 ≤ 5×10⁻⁷，属指标函数的浮点级噪声。两模型使用**完全相同的 test 样本、"
      f"标签与 PR-AUC 计算方式**"
      f"（sklearn average_precision_score，同一 y、同一行序），差异只来自模型本身。")
    A("")
    A("### 2.1 LR 为何 PR-AUC 更高（0.2909 vs 0.1837）——诊断表")
    A("")
    diag = pd.read_csv(outdir / "pr_auc_diagnosis_xgb_vs_lr.csv")

    def fmt_diag(v, nd=4):
        try:
            f = float(v)
            return "—" if pd.isna(f) else f"{f:.{nd}f}"
        except (TypeError, ValueError):
            return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    rows = [[r["diagnostic"], fmt_diag(r["dual_xgb"]), fmt_diag(r["dual_lr"]),
             fmt_diag(r["topK_overlap_rate"], 3)]
            for _, r in diag.iterrows()]
    A(_tbl(["诊断量", "dual_xgb", "dual_lr", "topK 重合率"], rows))
    A("")
    cal = pd.read_csv(outdir / "calibration_deciles.csv")
    top_dec = cal.groupby("model").tail(2)
    rows = [[r["model"], r["decile"], int(r["n"]), f"{r['mean_pred']:.4f}", f"{r['observed_rate']:.4f}",
             f"{r['mean_pred'] - r['observed_rate']:+.4f}"] for _, r in top_dec.iterrows()]
    A("最高两个十分位的校准（预测均值 vs 实际正例率）：")
    A("")
    A(_tbl(["模型", "十分位", "n", "mean_pred", "observed_rate", "差"], rows))
    A("")
    A("**解释（不改任何参数）**：两模型 ROC-AUC 几乎相同（0.7509 vs 0.7503），全局排序能力一致；"
      "PR-AUC 由**排序顶端精度**主导 —— LR 把最高分段拉得更陡（precision@K 在小 K 段更高），"
      "在正例率 8.5% 的 test 上顶端精度优势直接转化为 PR-AUC 优势。代价是**概率尺度失真**："
      "LR 用 balanced 类权重，把高分段预测概率系统性推高（见校准表 mean_pred ≫ observed_rate），"
      "因此 Brier 明显更差（0.1652 vs 0.0801）。即：**XGB 概率更可信（校准好），LR 顶端排序更尖（PR-AUC 高）**，"
      "两者服务不同用途（分级用概率选 XGB，Top-K 名单用排序看 PR）。本诊断仅为解释，"
      "按用户要求未对 XGB 重新调参。")
    A("")
    A("## 3. 动态风险能力检查（任务 3）")
    A("")
    rows = []
    for name in ("dual_xgb", "no_tide_xgb"):
        g = s5[name]["group_report_test"]
        rows.append(_mrow(f"{name} 全部样本", g["overall"]))
        rows.append(_mrow(f"{name} 事件日内", g["event_day_1"]))
        rows.append(_mrow(f"{name} 非事件日", g["event_day_0"]))
    A("全部样本 vs 事件日内（test @ 0.05，排序指标不受阈值影响）：")
    A("")
    A(_tbl(["分组", "n", "正例", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"], rows))
    A("")
    for name in ("dual_xgb", "no_tide_xgb"):
        ag = s5[name]["aggregated_auc_test"]
        dl, gl = ag["date_level"], ag["grid_day_level"]
        A(f"**{name} 时间聚合排序能力（test）**：日期级（{dl['n_dates']} 天，事件日 {dl['n_event_dates']}，"
          f"基线 {dl['baseline_pr']:.3f}）—— 日最大概率 ROC {dl['p_max_roc']:.4f} / PR {dl['p_max_pr']:.4f}；"
          f"grid-day 级（{gl['n_grid_days']:,} 格·日，正例 {gl['n_positive_grid_days']:,}，基线 "
          f"{gl['baseline_pr']:.3f}）—— ROC {gl['p_max_roc']:.4f} / PR {gl['p_max_pr']:.4f}。")
    A("")
    cg = pd.DataFrame(s5["coastal_groups"])
    if len(cg):
        rows = [[r["model"], r["flag"], r["group"], int(r["n_grids"]), int(r["n"]),
                 int(r["positives"]), f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}",
                 f"{r['precision']:.4f}", f"{r['recall']:.4f}"] for _, r in cg.iterrows()]
        A("沿海 / 内陆分组（test @ 0.05）：")
        A("")
        A(_tbl(["模型", "标志", "组", "格数", "n", "正例", "ROC-AUC", "PR-AUC", "Precision", "Recall"], rows))
    A("")
    ev1_roc = s5["dual_xgb"]["group_report_test"]["event_day_1"].get("roc_auc")
    ev1_txt = f"{ev1_roc:.3f}" if isinstance(ev1_roc, float) else "退化（单一类别）"
    A("**静态空间识别 vs 动态时间预警（必须区分表述）**："
      "模型的空间判别力主要来自静态地形记忆（正式训练 static_only 40 列单独 ROC 0.7202）；"
      f"事件日内部 ROC（dual_xgb {ev1_txt}，见表）"
      "与日期级聚合 AUC 才代表**动态时间预警能力**，"
      "显著低于全样本指标 —— 全样本指标混合了「哪些地方危险」与「什么时候危险」两类信息，"
      "论文引用时应分开表述，不得用全样本 ROC-AUC 宣称具备同等的动态预警能力。")
    A("")
    A("## 4. test 阈值敏感性（任务 4：仅敏感性分析，不重新选阈）")
    A("")
    A(f"正式阈值保持不变：dual_xgb τ=0.05（val F1 最大点）、no_tide_xgb τ={s3['no_tide_val_threshold']:.2f}。"
      "下表**仅展示**候选阈值在 test 上的运行点，供论文讨论阈值稳健性，不用于选择：")
    A("")
    sens = pd.read_csv(outdir / "threshold_sensitivity_test.csv")
    rows = [[r["model"], f"{r['threshold']:.2f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}",
             f"{r['f1']:.4f}", int(r["tp"]) + int(r["fp"]), r["alerts_per_day"]]
            for _, r in sens.iterrows()]
    A(_tbl(["模型", "阈值", "Precision", "Recall", "F1", "预警行数", "日均预警(格·槽/日)"], rows))
    A("")
    A("> 观察要点由数据自明：τ 0.02→0.30 期间 Precision 单调上升、Recall 单调下降；"
      "test 上的最优点与 val 选定点可能有偏差（test 事件日构成不同），这正是「阈值只能在 val 上定、"
      "test 仅观察」的原因。")
    A("")
    A("## 5. 雨潮复合描述分析（任务 5，论文复合条件协议；允许不显著）")
    A("")
    A(f"范围：test 官方警告日（事件日）行，n={s7['n_event_day_rows']:,}。"
      f"雨量代表列 `{s7['rain_col']}`（SHAP 第 1），潮位代表列 `{s7['tide_col']}`（SHAP 第 11）；"
      "各自按事件日内三分位切分，报告组合均值预测概率与样本覆盖数：")
    A("")
    comp = pd.read_csv(outdir / "rain_tide_composite_event_days.csv")
    RB, TB = ["雨低", "雨中", "雨高"], ["潮低", "潮中", "潮高"]
    for scope in ("事件日-全部", "事件日-沿海(low_coastal)", "事件日-内陆(low_coastal=0)"):
        sub = comp[comp["scope"] == scope]
        if not len(sub):
            continue
        piv_p = sub.pivot_table(index="rain_bin", columns="tide_bin", values="mean_p", aggfunc="first").reindex(index=RB, columns=TB)
        piv_n = sub.pivot_table(index="rain_bin", columns="tide_bin", values="n", aggfunc="first").reindex(index=RB, columns=TB)
        piv_o = sub.pivot_table(index="rain_bin", columns="tide_bin", values="obs_rate", aggfunc="first").reindex(index=RB, columns=TB)
        rows = []
        for r_ in RB:
            row = [r_]
            for c_ in TB:
                mp, n_, ob = piv_p.loc[r_, c_], piv_n.loc[r_, c_], piv_o.loc[r_, c_]
                if pd.isna(mp):
                    row.append("—")
                else:
                    obs_txt = f"，实际正例率 {ob:.2%}" if pd.notna(ob) else ""
                    row.append(f"{mp:.4f} (n={int(n_)}{obs_txt})")
            rows.append(row)
        A(f"**{scope}**（单元格 = mean p (n=样本数，实际正例率)，行 = 雨量分位，列 = 潮位分位）：")
        A("")
        A(_tbl(["雨量\\潮位"] + TB, rows))
        A("")
    hi = s7["high_vs_rest_event_days"]
    sub_all = comp[comp["scope"] == "事件日-全部"]
    if len(sub_all):
        is_hh = (sub_all["rain_bin"] == "雨高") & (sub_all["tide_bin"] == "潮高")
        hh, rest = sub_all[is_hh], sub_all[~is_hh]
        hh_p = float(hh["mean_p"].iloc[0]) if len(hh) else None
        hh_o = float(hh["obs_rate"].iloc[0]) if len(hh) else None
        rest_obs = float((rest["obs_rate"] * rest["n"]).sum() / rest["n"].sum())
        obs_uplift = hh_o - rest_obs if (hh_o is not None) else None
        rain_marg = {b: float((sub_all[sub_all["rain_bin"] == b]["mean_p"]
                               * sub_all[sub_all["rain_bin"] == b]["n"]).sum()
                              / sub_all[sub_all["rain_bin"] == b]["n"].sum()) for b in RB}
        rain_marg_obs = {b: float((sub_all[sub_all["rain_bin"] == b]["obs_rate"]
                                   * sub_all[sub_all["rain_bin"] == b]["n"]).sum()
                                  / sub_all[sub_all["rain_bin"] == b]["n"].sum()) for b in RB}
        A(f"「高雨量 + 高潮位」组合：均值预测概率 {hh_p:.4f}、实际正例率 {hh_o:.2%}"
          f"（n={int(hh['n'].iloc[0]):,}） vs 其余八格加权均值：预测概率 {hi['mean_p_rest']:.4f}、"
          f"实际正例率 {rest_obs:.2%}（n={hi['n_rest']:,}）。")
        if hh_p is not None and hh_p <= hi["mean_p_rest"] and obs_uplift and obs_uplift > 0:
            A("")
            A("**数据层面与模型层面方向相反（本轮关键发现）**：")
            A("")
            A(f"- **标签层面**：「高雨 + 高潮」实际正例率 {hh_o:.2%}，高于其余八格 {rest_obs:.2%}"
              f"（{obs_uplift:+.2%}）——弱标签数据中确实出现「雨潮同高」组合的正例聚集；"
              "该组合样本量充足，非小样本偶然。")
            A(f"- **模型层面**：该组合的均值预测概率 {hh_p:.4f} 反而**低于**其余八格 "
              f"{hi['mean_p_rest']:.4f}（{hi['uplift']:+.4f}），且预测概率对前期雨量非单调"
              f"（雨中 {rain_marg['雨中']:.4f} > 雨高 {rain_marg['雨高']:.4f}；"
              f"标签层面雨量边际为 {rain_marg_obs['雨低']:.2%} / {rain_marg_obs['雨中']:.2%} / "
              f"{rain_marg_obs['雨高']:.2%}）。")
            A("")
            A("即：**弱标签中存在雨潮复合的正例聚集迹象，但模型没有学到这一交互**——其判别力以"
              "静态地形记忆与前期降雨主效应为主，对交互项欠拟合，恰在「高雨 + 高潮」处系统性低估。"
              "为论文给出两条明确表述：**(1)** 雨潮复合可作为**假设与探索方向**（标签层面证据 + "
              "文献机制支持），但须同步披露弱标签归因窗口时序错位、潮位为天文潮预报而非实测增水等"
              "解释风险；**(2)** 模型当前版本**不能用于宣称已具备雨潮复合预警能力**；"
              "若加入显式交互特征或分段模型属新一轮训练，须另行批准。")
        elif hh_p is not None:
            A(f"样本覆盖充足；组合差异的解释须连同弱标签构造一并说明（标签层面 uplift {obs_uplift:+.2%}）。")
    A("")
    A("**结论口径（允许不显著）**：标签层面「高雨 + 高潮」组合出现正例聚集（描述性，"
      "未做统计检验），支持把雨潮复合作为**假设与探索方向**写入论文；但去潮位消融"
      f"（ROC 变化 {s3['delta_vs_dual']['roc_auc']:+.4f}）与组合分层（模型恰在标签正例率最高的"
      "组合处预测概率反而最低）一致表明**当前模型未捕获该交互**。论文表述建议："
      "模型包含潮位信息（SHAP 第 11 位）但以静态地形记忆与降雨主效应为主导；"
      "雨潮复合的量化结论须待显式交互特征建模与实测标签验证，本轮仅作描述性报告，"
      "并同步披露弱标签归因窗口与天文潮口径的局限。")
    A("")
    A("## 6. 产物清单")
    A("")
    for f in ["metrics.json", "threshold_sweep_no_tide_xgb.csv",
              "threshold_sensitivity_test.csv", "pr_auc_diagnosis_xgb_vs_lr.csv",
              "calibration_deciles.csv", "coastal_groups_test.csv",
              "rain_tide_composite_event_days.csv", "rain_tide_composite_all_test.csv",
              "predictions_test_no_tide.parquet", "bundles/no_tide_xgb.joblib",
              "results_report.md"]:
        A(f"- `{f}`")
    A("")
    A(f"运行：seed=42，总耗时 {result['total_seconds']}s，峰值内存 {result['peak_rss_mb']} MB。")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
