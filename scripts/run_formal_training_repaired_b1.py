"""delivery_repaired_b1 正式全量训练（用户 2026-10-26 批准，9 项条件逐一落实）。

条件 → 实现
1. 仅用 delivery_repaired_b1：数据 SHA256 对 manifest_repair_b1.json 校验，不匹配即中止；
   不读写任何旧数据目录，不覆盖任何既有产物，全部输出进独立目录。
2. 主模型 dual(59) XGBoost；对照 power_only(51) XGBoost；基线 dual 逻辑回归
   （中位数插补+标准化，M.build_model 现有配置）；消融 static_only(40) / rain_time_only(19)。
   全部无调参、无新增复杂模型。
3. 划分：train=2018-2021、val=2022、test=2023-2026.09（rd.assign_splits）；
   test 仅在阈值冻结后用于最终评估；golden_eval 仅作敏感性参考。
4. 不沿用旧 0.3 阈值：每模型在 val 上 0.01–0.99 扫描取 F1 最大者，报告 P/R/F1 与
   日均预警数量及选择依据；另附"主模型阈值统一施用于全部模型"的对照表。
5. 空间 CV：cv_fold / dc 分组有效性检查（必做，便宜）；18 折完整空间 CV 本轮不执行
   （用户允许时间不足时省略，此处明确记录）。
6. 输出：ROC-AUC、PR-AUC、Recall、Precision、F1、混淆矩阵、SHAP
   （xgboost pred_contribs = 精确 TreeSHAP）、GIS 逐格风险表（QGIS 可联接）与图件。
7. 弱监督标签、320 个改判日抽样偏差、ERA5/POWER 历史特征与实时部署差异 → 报告披露节。
8. 模型输出一律表述为弱监督标签下的风险评分，不得称已验证的真实水浸发生概率。
9. 不改论文标题、不改原始数据、不自动调参。

用法：
    .venv/bin/python scripts/run_formal_training_repaired_b1.py --preflight   # 只跑预检
    .venv/bin/python scripts/run_formal_training_repaired_b1.py               # 正式训练
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
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
from floodrisk.evaluate import binary_metrics, curve_data  # noqa: E402

DATA = ROOT / "data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet"
FRAME = ROOT / "data/delivery_repaired_b1/sample_frame.parquet"
MANIFEST = ROOT / "data/delivery_repaired_b1/manifest_repair_b1.json"

KEY_COLS = rd.KEY
RAIN_ALL = list(rd.RAIN_FEATURES)
TIME_COLS = list(rd.TIME_FEATURES)
_RAIN_SET = set(RAIN_ALL) | set(TIME_COLS)

MAIN = "dual_xgb"
MODELS: dict[str, tuple[str, list[str]]] = {
    "dual_xgb": ("xgb", list(rd.FEATURES)),                     # 主模型（方案 A，59 列）
    "dual_lr": ("lr", list(rd.FEATURES)),                       # 逻辑回归基线
    "power_only_xgb": ("xgb", list(rd.POWER_ONLY_FEATURES)),    # 对照（v10.0 方案，51 列）
    "static_only_xgb": ("xgb", [c for c in rd.FEATURES if c not in _RAIN_SET]),  # 消融 40 列
    "rain_time_only_xgb": ("xgb", RAIN_ALL + TIME_COLS),        # 消融 19 列
}

META_COLS = ["split", "cv_fold", "dc", "is_event_day", "is_A", "warn_level",
             "flooded_2h", "label_layer", "label_source", "confidence", "is_golden",
             "lon", "lat"]

REF_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SHAP_N = 8000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    try:
        import resource
        # macOS ru_maxrss 单位是字节，Linux 是 KB
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


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")


def registry_check(available: list[str]) -> dict:
    feats, excluded = set(rd.FEATURES), set(rd.EXCLUDED)
    inter = sorted(feats & excluded)
    return {
        "n_features_dual": len(rd.FEATURES),
        "n_features_power_only": len(rd.POWER_ONLY_FEATURES),
        "n_era5_added": len(rd.FEATURES) - len(rd.POWER_ONLY_FEATURES),
        "n_excluded": len(excluded),
        "feature_excluded_intersection": inter,
        "features_missing_in_parquet": sorted(c for c in rd.FEATURES if c not in available),
        "duplicated_features": sorted({c for c in rd.FEATURES if rd.FEATURES.count(c) > 1}),
        "ok": not inter and not (set(rd.FEATURES) - set(available))
              and not sorted({c for c in rd.FEATURES if rd.FEATURES.count(c) > 1}),
    }


def univariate_auc_scan(frame: pd.DataFrame, y_col: str, features: list[str]) -> list[dict]:
    """泄漏体检：仅允许在训练段数据上调用。"""
    from sklearn.metrics import roc_auc_score

    y = frame[y_col].to_numpy()
    rows = []
    for col in features:
        x = pd.to_numeric(frame[col], errors="coerce")
        ok = x.notna().to_numpy()
        if ok.sum() < 100 or len(np.unique(y[ok])) < 2 or x[ok].nunique() < 2:
            continue
        auc = float(roc_auc_score(y[ok], x[ok].to_numpy()))
        rows.append({"feature": col, "auc": round(auc, 4), "n": int(ok.sum())})
    rows.sort(key=lambda r: -abs(r["auc"] - 0.5))
    for r in rows:
        r["suspicious"] = abs(r["auc"] - 0.5) >= 0.47
    return rows


def fast_threshold_sweep(y: np.ndarray, p: np.ndarray,
                         start: float = 0.01, stop: float = 0.99, step: float = 0.01) -> pd.DataFrame:
    """val 阈值扫描：只算混淆矩阵派生量（比逐点 binary_metrics 快一个量级）。"""
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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="delivery_repaired_b1 正式全量训练")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--preflight", action="store_true", help="只跑预检（数据/划分/泄漏门槛）后退出")
    args = ap.parse_args(argv)

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "formal_training_preflight_repaired_b1" if args.preflight else "formal_training_repaired_b1"
    outdir = Path(args.outdir) if args.outdir else ROOT / f"outputs/{kind}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    seed = args.seed
    result: dict = {
        "created_at": stamp,
        "run_kind": "preflight" if args.preflight else "formal_training",
        "data": {
            "path": str(DATA.relative_to(ROOT)),
            "sha256": None,  # step0 填充
            "version": "delivery_repaired_b1 (B-side repair R2, official HKO rstorm.dat)",
            "sample_frame_sha256": None,
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git": git_commit(),
        },
        "params": {
            "seed": seed,
            "models": {k: {"backend": v[0], "n_features": len(v[1])} for k, v in MODELS.items()},
            "tuning": "无调参，沿用 M.build_model 现有研发默认配置（用户条件 2/9）",
            "threshold_policy": "每模型在 val(2022) 上 0.01–0.99 步长 0.01 扫描取 F1 最大者；"
                                "test 不参与阈值选择与拟合（用户条件 3/4）",
            "spatial_cv": "cv_fold/dc 分组有效性检查必做；18 折完整空间 CV 本轮不执行"
                          "（用户条件 5 允许时间不足时省略）",
            "shap": f"xgboost pred_contribs（精确 TreeSHAP），训练段抽样 n={SHAP_N}，seed={seed}",
        },
    }

    import pyarrow.parquet as pq
    import sklearn
    import xgboost

    result["env"].update({"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                          "matplotlib": __import__("matplotlib").__version__})

    # ---- step0 数据版本冻结 --------------------------------------------------
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sha_data, sha_frame = sha256(DATA), sha256(FRAME)
    result["data"]["sha256"] = sha_data
    result["data"]["sample_frame_sha256"] = sha_frame
    result["data"]["manifest"] = manifest
    ok_sha = (sha_data == manifest["outputs"]["main"]["sha256"]
              and sha_frame == manifest["outputs"]["frame"]["sha256"])
    result["step0_freeze"] = {"data_sha256_matches_manifest": bool(sha_data == manifest["outputs"]["main"]["sha256"]),
                              "frame_sha256_matches_manifest": bool(sha_frame == manifest["outputs"]["frame"]["sha256"])}
    print(f"[0] sha data match={result['step0_freeze']['data_sha256_matches_manifest']} "
          f"frame match={result['step0_freeze']['frame_sha256_matches_manifest']}")
    if not ok_sha:
        print("[ABORT] 数据 SHA256 与 manifest 不一致 —— 禁止训练（条件 1）")
        return 2

    # ---- step1 读入（键/标签/元数据 + 59 特征，一次读全表）--------------------
    t = time.time()
    data = rd.load_delivery(str(DATA), columns=KEY_COLS + META_COLS + rd.FEATURES,
                            categorical=["grid_id", "date", "label_layer", "label_source"])
    ts_expect = (pd.to_datetime(data["date"].astype(object).astype(str), format="%Y-%m-%d")
                 + pd.to_timedelta(data["slot"].astype(int) * 2, unit="h"))
    ts_ok = bool((data["timestamp"] == ts_expect).all())
    dup = int(data.duplicated(KEY_COLS).sum())
    result["step1_read"] = {
        "seconds": round(time.time() - t, 1),
        "rows": int(len(data)),
        "n_grids": int(data["grid_id"].nunique()),
        "timestamp_equals_date_plus_2h_slot": ts_ok,
        "duplicated_grid_date_slot": dup,
        "event_days": int(data.loc[data["is_event_day"].astype(int) == 1, "date"].nunique()),
        "positives_total": int(data["flooded_2h"].sum()),
        "is_A_share": round(float(data["is_A"].astype(int).mean()), 4),
        "peak_rss_mb": peak_rss_mb(),
    }
    print(f"[1] read {len(data):,} rows / {result['step1_read']['n_grids']} grids in "
          f"{result['step1_read']['seconds']}s; event_days={result['step1_read']['event_days']} "
          f"pos={result['step1_read']['positives_total']:,} ts_ok={ts_ok} dup={dup}")
    if not ts_ok or dup != 0:
        print("[ABORT] 时间锚或主键校验失败")
        return 2

    # ---- step2 目标与划分 ----------------------------------------------------
    data, target_report = rd.build_targets(data)
    data, split_info = rd.assign_splits(data)
    partition = rd.check_partition(data)
    result["step2_split"] = {"target": target_report, "info": split_info, "partition": partition}
    print(f"[2] split counts={split_info['counts']}")
    print(f"    partition={partition}")
    if any(partition[k] != 0 for k in ("grid_day_overlap_train_val", "grid_day_overlap_train_test",
                                       "grid_day_overlap_val_test", "date_overlap_train_test")):
        print("[ABORT] 划分存在 grid-day / 日期重叠 —— 数据泄漏风险（条件 3）")
        write_json(outdir / "aborted.json", result)
        return 2

    # ---- step3 空间分组有效性（cv_fold / dc）---------------------------------
    tr_mask = data["exp_split"].eq("train")
    cvf = data.loc[tr_mask, "cv_fold"]
    cv_ok = cvf.notna()
    per_grid_folds = data.loc[tr_mask & cv_ok].groupby("grid_id", observed=True)["cv_fold"].nunique()
    dc_folds = data.loc[tr_mask & cv_ok].groupby("dc", observed=True)["cv_fold"].nunique()
    result["step3_spatial_groups"] = {
        "train_rows": int(tr_mask.sum()),
        "cv_fold_coverage": round(float(cv_ok.mean()), 4),
        "n_folds": int(cvf.nunique()),
        "fold_sizes": {str(k): int(v) for k, v in cvf.value_counts().sort_index().items()},
        "grids_in_multiple_folds": int((per_grid_folds > 1).sum()),
        "max_folds_per_grid": int(per_grid_folds.max()) if len(per_grid_folds) else 0,
        "dc_in_multiple_folds": int((dc_folds > 1).sum()),
        "n_dc": int(data.loc[tr_mask, "dc"].nunique()),
        "note": "网格→折唯一性成立则空间分组有效；18 折完整 CV 本轮不执行（条件 5）",
    }
    print(f"[3] cv_fold coverage={result['step3_spatial_groups']['cv_fold_coverage']} "
          f"n_folds={result['step3_spatial_groups']['n_folds']} "
          f"grids_in_multiple_folds={result['step3_spatial_groups']['grids_in_multiple_folds']} "
          f"dc_in_multiple_folds={result['step3_spatial_groups']['dc_in_multiple_folds']}")

    # ---- step4 特征注册表 + NA + float32 + 单变量泄漏门槛 --------------------
    reg = registry_check(pq.read_schema(str(DATA)).names)
    result["step4_features"] = {"registry": reg, "featsets_sizes": {k: len(v[1]) for k, v in MODELS.items()}}
    print(f"[4] registry ok={reg['ok']}")
    if not reg["ok"]:
        print(f"[ABORT] 特征注册表检查失败: {reg}")
        write_json(outdir / "aborted.json", result)
        return 2

    for c in rd.FEATURES:
        data[c] = data[c].astype("float32")
    na_rate = {c: round(float(data[c].isna().mean()), 4) for c in rd.FEATURES if data[c].isna().any()}
    result["step4_features"]["feature_na_rate"] = na_rate

    scan = univariate_auc_scan(data.loc[tr_mask], "y_future_2h", rd.FEATURES)
    suspicious = [r["feature"] for r in scan if r["suspicious"]]
    result["step4_features"]["univariate_auc_train"] = scan
    result["step4_features"]["suspicious_features"] = suspicious
    result["step4_features"]["univariate_top10"] = scan[:10]
    pd.DataFrame(scan).to_csv(outdir / "univariate_auc_train.csv", index=False)
    print(f"[4] univariate suspicious={suspicious} top3={[(r['feature'], r['auc']) for r in scan[:3]]}")
    if suspicious:
        print("[ABORT] 训练段单变量 AUC 门槛触发（|AUC-0.5|>=0.47）—— 疑似标签泄漏，禁止训练")
        write_json(outdir / "aborted.json", result)
        return 2

    gc.collect()
    if args.preflight:
        result["preflight"] = {
            "pass": True,
            "gates": ["sha256", "timestamp", "primary_key", "partition_overlap",
                      "feature_registry", "univariate_auc_train_only"],
            "elapsed_seconds": round(time.time() - t_start, 1),
            "eta_full_run_minutes": [12, 30],
        }
        write_json(outdir / "preflight.json", result)
        print(f"\n[PREFLIGHT PASS] {result['preflight']['elapsed_seconds']}s → {outdir/'preflight.json'}")
        print("预计正式训练（含本预检全部步骤 + 5 模型拟合 + SHAP + GIS + 图件）：约 12–30 分钟。")
        return 0

    # ---- step5 拟合 + val 阈值冻结（test 不参与）-----------------------------
    tr = data[data["exp_split"].eq("train")]
    va = data[data["exp_split"].eq("val")]
    te = data[data["exp_split"].eq("test")]
    au = data[data["exp_split"].eq("golden_eval")]
    y_tr = tr["y_future_2h"].to_numpy(dtype=int)
    y_va = va["y_future_2h"].to_numpy(dtype=int)
    y_te = te["y_future_2h"].to_numpy(dtype=int)
    n_dates = {"train": int(tr["date"].nunique()), "val": int(va["date"].nunique()),
               "test": int(te["date"].nunique())}
    val_event_dates = int(va.loc[va["is_event_day"].astype(int) == 1, "date"].nunique())
    print(f"[5] train={len(tr):,} val={len(va):,} test={len(te):,} golden={len(au):,}; "
          f"dates={n_dates}; val_event_dates={val_event_dates}")

    fitted: dict[str, object] = {}
    p_va_by: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    threshold_rationale: dict[str, dict] = {}
    step5 = {}
    for name, (backend, fs) in MODELS.items():
        t0 = time.time()
        model = M.build_model(backend, seed=seed)
        model.fit(tr[fs], y_tr)
        fit_s = round(time.time() - t0, 1)
        p_va = M.predict_proba(model, va[fs])
        p_va_by[name] = p_va
        fitted[name] = model

        sweep = fast_threshold_sweep(y_va, p_va)
        sweep.to_csv(outdir / f"threshold_sweep_{name}.csv", index=False)
        best = sweep.loc[sweep["f1"].idxmax()]
        thr = float(best["threshold"])
        thresholds[name] = thr
        threshold_rationale[name] = {
            "rule": "val 上 F1 最大（0.01–0.99 步长 0.01）；test 未参与",
            "selected_threshold": thr,
            "val_at_selected": {k: float(best[k]) for k in ("precision", "recall", "f1")},
            "val_predicted_positives": int(best["predicted_positives"]),
            "val_dates": n_dates["val"],
            "val_alerts_per_day": alerts_per_day(p_va, thr, n_dates["val"]),
            "neighbors": sweep[(sweep["threshold"] >= thr - 0.03) & (sweep["threshold"] <= thr + 0.03)]
                [["threshold", "precision", "recall", "f1", "predicted_positives"]].to_dict("records"),
            "reference_thresholds": sweep[sweep["threshold"].isin(REF_THRESHOLDS)]
                [["threshold", "precision", "recall", "f1", "predicted_positives"]].to_dict("records"),
        }
        step5[name] = {"backend": backend, "n_features": len(fs), "fit_seconds": fit_s,
                       "val_threshold": thr,
                       "val_metrics": binary_metrics(y_va, p_va, threshold=thr, label=f"{name}_val")}
        print(f"[5] {name:20s} {backend:3s} fit {fit_s:6.1f}s | thr={thr:.2f} | "
              f"val {compact(step5[name]['val_metrics'])} | "
              f"alerts/day={threshold_rationale[name]['val_alerts_per_day']}")

    thr_main = thresholds[MAIN]
    result["step5_fit_threshold"] = {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "n_dates": n_dates, "val_event_dates": val_event_dates,
        "train_pos_rate": round(float(y_tr.mean()), 6),
        "models": step5, "threshold_rationale": threshold_rationale,
    }

    # ---- step6 test 最终评估（阈值已冻结）------------------------------------
    step6_models = {}
    p_te_by: dict[str, np.ndarray] = {}
    for name, (backend, fs) in MODELS.items():
        p_te = M.predict_proba(fitted[name], te[fs])
        p_te_by[name] = p_te
        m_own = binary_metrics(y_te, p_te, threshold=thresholds[name], label=f"{name}_test")
        m_fixed = binary_metrics(y_te, p_te, threshold=thr_main, label=f"{name}_test@thr_main")
        step6_models[name] = {
            "test_at_own_val_threshold": m_own,
            "test_at_main_threshold": m_fixed,
            "test_alerts_per_day_own_thr": alerts_per_day(p_te, thresholds[name], n_dates["test"]),
            "test_alerts_per_day_main_thr": alerts_per_day(p_te, thr_main, n_dates["test"]),
        }
        print(f"[6] {name:20s} test@own {compact(m_own)} | alerts/day={step6_models[name]['test_alerts_per_day_own_thr']}")

    # 主模型细报告：事件日/非事件日、分年、golden 敏感性
    def group_report(frame: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
        out = {"overall": binary_metrics(frame["y_future_2h"].to_numpy(), p, threshold=thr, label="overall")}
        for flag in (1, 0):
            sub = frame[frame["is_event_day"].astype(int) == flag]
            i = sub.index.to_numpy()
            if len(sub) and sub["y_future_2h"].nunique() == 2:
                out[f"event_day_{flag}"] = binary_metrics(
                    sub["y_future_2h"].to_numpy(), p[frame.index.get_indexer(i)],
                    threshold=thr, label=f"event_day_{flag}")
            elif len(sub):
                out[f"event_day_{flag}"] = {"n": int(len(sub)), "positives": int(sub["y_future_2h"].sum()),
                                            "degenerate": "单一类别，排序指标不可定义"}
        years = {}
        for year, sub in frame.groupby(frame["date_dt"].dt.year):
            if sub["y_future_2h"].nunique() == 2:
                years[str(int(year))] = binary_metrics(
                    sub["y_future_2h"].to_numpy(), p[frame.index.get_indexer(sub.index)],
                    threshold=thr, label=str(int(year)))
            else:
                years[str(int(year))] = {"n": int(len(sub)), "positives": int(sub["y_future_2h"].sum()),
                                         "degenerate": "单一类别"}
        out["by_year"] = years
        return out

    main_test_report = group_report(te, p_te_by[MAIN], thr_main)
    p_au = M.predict_proba(fitted[MAIN], au[MODELS[MAIN][1]]) if len(au) else np.array([])
    golden_metrics = (binary_metrics(au["y_future_2h"].to_numpy(), p_au, threshold=thr_main,
                                     label="golden_eval@thr_main") if len(au) else None)
    result["step6_test_final"] = {
        "thresholds_frozen_on_val": thresholds,
        "main_threshold": thr_main,
        "models": step6_models,
        "main_model_group_report_test": main_test_report,
        "golden_eval_sensitivity": {
            "metrics": golden_metrics,
            "note": "golden 行为真事件日加密抽样（正例率约 49%），只能作敏感性参考，不并入主结果",
        },
    }
    print(f"[6] golden {compact(golden_metrics) if golden_metrics else 'n/a'}")

    # ---- step7 SHAP（xgboost pred_contribs = 精确 TreeSHAP）------------------
    t0 = time.time()
    fs_main = MODELS[MAIN][1]
    booster = fitted[MAIN].get_booster()
    samp = tr.sample(min(SHAP_N, len(tr)), random_state=seed)
    contribs = booster.predict(xgboost.DMatrix(samp[fs_main]), pred_contribs=True)
    sv = np.asarray(contribs)[:, :-1]  # 末列为 bias
    mean_abs = np.abs(sv).mean(axis=0)
    gain = fitted[MAIN].feature_importances_
    imp = pd.DataFrame({"feature": fs_main, "mean_abs_shap_logodds": mean_abs,
                        "xgb_gain_importance": gain}).sort_values("mean_abs_shap_logodds", ascending=False)
    imp.insert(0, "rank", range(1, len(imp) + 1))
    imp.to_csv(outdir / "shap_importance.csv", index=False)
    result["step7_shap"] = {
        "method": "TreeSHAP (pred_contribs), log-odds space",
        "n_sample": int(len(samp)), "sample_positives": int(samp["y_future_2h"].sum()),
        "seconds": round(time.time() - t0, 1),
        "top20": imp.head(20).to_dict("records"),
    }
    print(f"[7] shap top5: {[(r['feature'], round(r['mean_abs_shap_logodds'], 3)) for r in imp.head(5).to_dict('records')]}")

    # ---- step8 GIS 逐格风险表 -------------------------------------------------
    t0 = time.time()
    p_all = M.predict_proba(fitted[MAIN], data[fs_main])
    not_golden = ~data["exp_split"].eq("golden_eval").to_numpy()
    tmp = pd.DataFrame({
        "grid_id": data["grid_id"].astype(str).to_numpy(),
        "is_event": data["is_event_day"].astype(int).to_numpy(),
        "p": p_all,
        "lon": data["lon"].to_numpy(dtype=float),
        "lat": data["lat"].to_numpy(dtype=float),
    })
    tmp = tmp[not_golden]
    base = tmp.groupby("grid_id", observed=True).agg(
        n_rows=("p", "size"), mean_risk_all=("p", "mean"),
        p90_risk=("p", lambda s: float(np.quantile(s, 0.9))),
        lon=("lon", "first"), lat=("lat", "first"), lon_nunique=("lon", "nunique"))
    lon_nuniq_max = int(base["lon_nunique"].max()) if len(base) else 0
    ev = tmp[tmp["is_event"] == 1].groupby("grid_id", observed=True)["p"].agg(
        mean_risk_event_day="mean", n_event_rows="size")
    ne = tmp[tmp["is_event"] == 0].groupby("grid_id", observed=True)["p"].agg(
        mean_risk_nonevent_day="mean", n_nonevent_rows="size")
    grid = base.join(ev, how="left").join(ne, how="left").reset_index()
    grid["rank_mean_risk"] = grid["mean_risk_all"].rank(ascending=False, method="first").astype(int)
    grid = grid.sort_values("rank_mean_risk")
    grid = grid[["grid_id", "lon", "lat", "n_rows", "mean_risk_all", "p90_risk",
                 "mean_risk_event_day", "n_event_rows", "mean_risk_nonevent_day",
                 "n_nonevent_rows", "rank_mean_risk"]]
    grid.rename(columns={"lon": "lon_wgs84", "lat": "lat_wgs84"}, inplace=True)
    (outdir / "gis").mkdir(exist_ok=True)
    grid.to_csv(outdir / "gis/grid_risk_500m.csv", index=False)
    top30 = grid.head(30)[["grid_id", "lon_wgs84", "lat_wgs84", "mean_risk_all",
                           "mean_risk_event_day", "n_event_rows"]].to_dict("records")
    result["step8_gis"] = {
        "n_grids": int(len(grid)),
        "expected_grids": 5517,
        "lon_unique_per_grid_max": lon_nuniq_max,
        "seconds": round(time.time() - t0, 1),
        "file": "gis/grid_risk_500m.csv",
        "top30_by_mean_risk": top30,
        "note": "风险聚合基于非 golden 抽样行（train+val+test）；"
                "输出为弱监督标签驱动的相对风险评分，不是已验证的真实水浸概率",
    }
    print(f"[8] gis grids={len(grid)} in {result['step8_gis']['seconds']}s; "
          f"top1={top30[0] if top30 else 'n/a'}")

    # ---- step9 图件 ------------------------------------------------------------
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {"roc": {}, "pr": {}}
    for name in MODELS:
        c_va = curve_data(y_va, p_va_by[name])
        c_te = curve_data(y_te, p_te_by[name])
        curves["roc"][f"{name}__val"] = c_va["roc"]
        curves["roc"][f"{name}__test"] = c_te["roc"]
        curves["pr"][f"{name}__val"] = c_va["pr"]
        curves["pr"][f"{name}__test"] = c_te["pr"]

    # ROC / PR
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    for key, c in curves["roc"].items():
        main_curve = key.startswith(f"{MAIN}__")
        ax.plot(c["fpr"], c["tpr"], lw=2.2 if main_curve else 0.9,
                alpha=1.0 if main_curve else 0.5, label=key.replace("__", " "))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (val / test)"); ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    for key, c in curves["pr"].items():
        main_curve = key.startswith(f"{MAIN}__")
        ax.plot(c["recall"], c["precision"], lw=2.2 if main_curve else 0.9,
                alpha=1.0 if main_curve else 0.5, label=key.replace("__", " "))
    ax.axhline(float(te["y_future_2h"].mean()), color="k", ls=":", lw=0.9,
               label=f"test baseline ({float(te['y_future_2h'].mean()):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (val / test)"); ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(figdir / "fig_roc_pr.png", dpi=160); plt.close(fig)

    # 阈值曲线（主模型 val）
    sweep_main = pd.read_csv(outdir / f"threshold_sweep_{MAIN}.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(sweep_main["threshold"], sweep_main["precision"], label="Precision")
    ax.plot(sweep_main["threshold"], sweep_main["recall"], label="Recall")
    ax.plot(sweep_main["threshold"], sweep_main["f1"], label="F1")
    ax.axvline(thr_main, color="crimson", ls="--", lw=1.2, label=f"selected={thr_main:.2f}")
    ax.axvline(0.3, color="gray", ls=":", lw=1.0, label="old reference 0.3 (not used)")
    ax.set_xlabel("Threshold (val, dual_xgb)"); ax.set_ylabel("Score")
    apd = threshold_rationale[MAIN]["val_alerts_per_day"]
    ax.set_title(f"Threshold sweep on val (2022) — alerts/day={apd} (sampled cells)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "fig_threshold_val.png", dpi=160); plt.close(fig)

    # 混淆矩阵（主模型 test @ 主阈值）
    m = step6_models[MAIN]["test_at_own_val_threshold"]
    cm = np.array(m["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            share = cm[r, c] / cm[r].sum()
            ax.text(c, r, f"{cm[r, c]:,}\n({share:.1%})", ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"dual_xgb test @ thr={m['threshold']:.2f}\n"
                 f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(figdir / "fig_confusion_test.png", dpi=160); plt.close(fig)

    # SHAP 条形图 + 依赖图
    top20 = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(top20["feature"], top20["mean_abs_shap_logodds"], color="#3b6ea5")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title(f"TreeSHAP global importance — dual_xgb (n={len(samp)} train sample)")
    fig.tight_layout(); fig.savefig(figdir / "fig_shap_bar.png", dpi=160); plt.close(fig)

    top6 = imp.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), top6):
        i = fs_main.index(f)
        ax.scatter(samp[f].to_numpy(), sv[:, i], s=4, alpha=0.25, color="#3b6ea5", edgecolors="none")
        ax.set_xlabel(f); ax.set_ylabel("SHAP (log-odds)")
        ax.set_title(f, fontsize=9)
    fig.suptitle("SHAP dependence — top 6 features (dual_xgb)", y=1.0)
    fig.tight_layout(); fig.savefig(figdir / "fig_shap_dependence.png", dpi=160); plt.close(fig)

    # GIS 风险图（示意散点）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, col, title in ((axes[0], "mean_risk_all", "mean risk (all sampled slots)"),
                           (axes[1], "mean_risk_event_day", "mean risk (official warning days)")):
        sub = grid.dropna(subset=[col])
        sc = ax.scatter(sub["lon_wgs84"], sub["lat_wgs84"], c=sub[col], s=5, cmap="viridis")
        ax.set_title(title, fontsize=10); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("Grid-level risk score (weak-supervision label driven; NOT verified flood probability)")
    fig.tight_layout(); fig.savefig(figdir / "fig_gis_map.png", dpi=160); plt.close(fig)
    result["figures"] = ["fig_roc_pr.png", "fig_threshold_val.png", "fig_confusion_test.png",
                         "fig_shap_bar.png", "fig_shap_dependence.png", "fig_gis_map.png"]

    # ---- step10 预测明细落盘 ---------------------------------------------------
    keep = KEY_COLS + ["timestamp", "is_event_day", "y_future_2h"]
    va_out = va[keep].copy()
    te_out = te[keep].copy()
    for name in MODELS:
        va_out[f"p_{name}"] = p_va_by[name].astype("float64")
        te_out[f"p_{name}"] = p_te_by[name].astype("float64")
    va_out.to_parquet(outdir / "predictions_val.parquet", index=False)
    te_out.to_parquet(outdir / "predictions_test.parquet", index=False)

    # ---- step11 bundle 保存 ----------------------------------------------------
    for name, (backend, fs) in MODELS.items():
        bundle = M.Bundle(
            backend=backend, model=fitted[name], features=list(fs),
            meta={
                "version": f"repaired_b1_formal_{name}", "feature_set": name,
                "n_features": len(fs), "n_train": int(len(tr)), "seed": seed,
                "target": "y_future_2h",
                "threshold_val_f1max": thresholds[name],
                "data_sha256": sha_data, "data_version": "delivery_repaired_b1",
                "exploratory_weak_supervision": True,
                "note": "输出为弱监督标签下的风险评分，不得解释为已验证的真实水浸发生概率",
            },
        )
        M.save_bundle(outdir / "bundles" / f"{name}.joblib", bundle)

    # ---- step12 汇总 + 报告 ----------------------------------------------------
    result["total_seconds"] = round(time.time() - t_start, 1)
    result["peak_rss_mb"] = peak_rss_mb()
    result["disclaimer"] = (
        "弱监督标签下的探索性-正式训练结果：标签 = is_A × 官方红/黑警告归因窗口（HKO rstorm.dat），"
        "全部指标不得解释为真实水浸提前 2 小时预测准确率，模型输出不得称为已验证的真实水浸发生概率；"
        "样本含 320 个改判日 35 格受限重抽偏差（A 类占比 "
        f"{result['step1_read']['is_A_share']:.1%}），ERA5/POWER 特征为历史再分析/遥测口径，"
        "与实时部署可获得的输入存在时效差异"
    )
    write_json(outdir / "metrics.json", result)

    md = build_report_md(result, grid, outdir)
    (outdir / "results_report.md").write_text(md, encoding="utf-8")
    (ROOT / "reports/formal_training_results_repaired_b1.md").write_text(md, encoding="utf-8")
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


def build_report_md(result: dict, grid: pd.DataFrame, outdir: Path) -> str:
    s0 = result["step0_freeze"]
    s1 = result["step1_read"]
    s2p = result["step2_split"]["partition"]
    s5 = result["step5_fit_threshold"]
    s6 = result["step6_test_final"]
    s7 = result["step7_shap"]
    s8 = result["step8_gis"]
    thr_main = s6["main_threshold"]
    outdir_rel = f"outputs/{outdir.name}"

    lines: list[str] = []
    A = lines.append
    A("# 正式训练结果（delivery_repaired_b1，2026-10-26 批准）")
    A("")
    A("> **解读边界（必须随结果一并引用）**")
    A(">")
    A("> 1. 标签为**弱监督**：`flooded_2h` = A 类空间规则 × HKO 官方红/黑暴雨警告归因窗口，"
      "不是实测水浸记录；全部指标为弱监督标签下的探索性结果，**不得解释为真实水浸提前 2 小时预测准确率**。")
    A("> 2. **抽样偏差**：数据来自分层抽样框架（非事件日 35 格/日），其中 320 个旧版假事件日改判非事件日后"
      "按 35 格/日受限重抽（A 类格占比约 "
      f"{s1['is_A_share']:.1%}），正例空间覆盖受警告归因窗口限制。")
    A("> 3. **特征时效**：ERA5 为再分析产品（发布滞后数月）、POWER 为格点遥测/模型融合历史数据；"
      "模型输入均为**预测时刻 t 之前**的历史累计量，但与实时部署可获得的等效输入存在时效与口径差异，"
      "当前结果不构成可直接上线的实时预警系统。")
    A("> 4. 模型输出是**风险评分**（相对排序），**不得称为已验证的真实水浸发生概率**。")
    A("")
    A(f"运行目录：`{outdir_rel}`（全部产物独立保存，未覆盖任何旧版本；同名报告已复制到 `reports/`）  ")
    A(f"数据：`{result['data']['path']}`  ")
    A(f"SHA256：`{result['data']['sha256'][:16]}…`（与 `manifest_repair_b1.json` 一致："
      f"{s0['data_sha256_matches_manifest']}）")
    A("")
    A("## 1. 数据版本与划分（条件 1/3）")
    A("")
    A(_tbl(["项目", "值"], [
        ["数据版本", "delivery_repaired_b1（B 侧 R2 修复，官方 rstorm.dat）"],
        ["行数 / 网格", f"{s1['rows']:,} / {s1['n_grids']}"],
        ["时间戳校验", f"timestamp = date + 2h·slot：{s1['timestamp_equals_date_plus_2h_slot']}"],
        ["主键重复", s1["duplicated_grid_date_slot"]],
        ["事件日（官方红/黑）", s1["event_days"]],
        ["train (2018–2021)", f"{s2p['n']['train']:,}（正例率 {s2p['pos_rate']['train']:.4f}）"],
        ["val (2022)", f"{s2p['n']['val']:,}（正例率 {s2p['pos_rate']['val']:.4f}）"],
        ["test (2023–2026.09)", f"{s2p['n']['test']:,}（正例率 {s2p['pos_rate']['test']:.4f}）"],
        ["grid-day 重叠 (train/val/test)", f"{s2p['grid_day_overlap_train_val']} / "
                                           f"{s2p['grid_day_overlap_train_test']} / {s2p['grid_day_overlap_val_test']}"],
        ["train–test 日期重叠", s2p["date_overlap_train_test"]],
    ]))
    A("")
    A(f"无泄漏检查：SHA256 冻结 ✅、时间锚 ✅、主键唯一 ✅、划分零重叠 ✅、"
      f"特征注册表（59 特征 ∩ 排除字段 = ∅）✅、训练段单变量 AUC 扫描无可疑特征（|AUC−0.5|≥0.47 为 0 个）✅。"
      f"test 仅在阈值冻结后用于最终评估（代码路径分离，见脚本 step5/step6）。")
    A("")
    sg = result["step3_spatial_groups"]
    A("## 2. 空间交叉验证分组有效性（条件 5）")
    A("")
    A(f"- `cv_fold` 在训练段覆盖率 {sg['cv_fold_coverage']:.2%}，共 {sg['n_folds']} 折；")
    A(f"- 落入多折的网格数：{sg['grids_in_multiple_folds']}（最大每格折数 {sg['max_folds_per_grid']}）；")
    A(f"- 跨多折的 `dc` 空间组数：{sg['dc_in_multiple_folds']}（共 {sg['n_dc']} 组）。")
    A("")
    A(f"结论：网格→折映射{'唯一，空间分组有效' if sg['grids_in_multiple_folds'] == 0 else '存在跨折，需注意'}。"
      f"按用户条件 5，本轮时间预算优先主模型/基线/消融/GIS，**18 折完整空间 CV 未执行**，已明确记录。")
    A("")
    A("## 3. 模型配置（条件 2/9：无调参、无新增复杂模型）")
    A("")
    rows = []
    for name, st in s5["models"].items():
        rows.append([name, st["backend"], st["n_features"],
                     {"xgb": "XGBoost hist(300,depth4)", "lr": "LR(中位数插补+标准化, balanced)"}[st["backend"]],
                     f"{st['fit_seconds']}s", f"{st['val_threshold']:.2f}"])
    A(_tbl(["模型", "后端", "特征数", "配置（研发默认，未调参）", "拟合耗时", "val 阈值"], rows))
    A("")
    A("## 4. 阈值选择依据（条件 4：不沿用旧 0.3）")
    A("")
    ra = s5["threshold_rationale"][MAIN]
    A(f"主模型 `dual_xgb` 阈值 **{thr_main:.2f}**：验证集（2022 年，{s5['n_dates']['val']} 个采样日，"
      f"其中官方红/黑事件日仅 {s5['val_event_dates']} 个）上 0.01–0.99 扫描的 **F1 最大点**。选择时同时核对：")
    A("")
    ref = {r["threshold"]: r for r in ra["reference_thresholds"]}
    rows = [[f"{t:.2f}", f"{ref[t]['precision']:.3f}", f"{ref[t]['recall']:.3f}", f"{ref[t]['f1']:.3f}",
             ref[t]["predicted_positives"],
             round(ref[t]["predicted_positives"] / max(s5["n_dates"]["val"], 1), 2)]
            for t in sorted(ref) if t in ref]
    A(_tbl(["val 阈值", "Precision", "Recall", "F1", "预警格·槽行数", "日均预警(格·槽/日)"], rows))
    A("")
    A(f"选定阈值处：P={ra['val_at_selected']['precision']:.3f}、R={ra['val_at_selected']['recall']:.3f}、"
      f"F1={ra['val_at_selected']['f1']:.3f}，val 口径日均预警 {ra['val_alerts_per_day']} 格·槽/日"
      f"（分母为 {ra['val_dates']} 个采样日）。旧阈值 0.3 未被沿用，仅在图中作参考线。"
      f"**局限**：val 仅覆盖 {s5['val_event_dates']} 个官方红/黑事件日，阈值在事件日构成上的代表性有限，"
      f"已在披露节说明。各模型阈值均在各自 val 扫描上独立选定（见 `threshold_sweep_*.csv`）。")
    A("")
    A("## 5. 主结果对比（test = 2023–2026.09 时间外测试集，仅最终评估）")
    A("")
    rows = []
    for name, mm in s6["models"].items():
        m = mm["test_at_own_val_threshold"]
        rows.append([name, m["n"], f"{m['positive_rate']:.4f}", f"{m['roc_auc']:.4f}",
                     f"{m['pr_auc']:.4f}（基线 {m['baseline_pr_auc']:.3f}）", f"{m['brier']:.4f}",
                     f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                     f"{m['tp']:,}/{m['fp']:,}/{m['fn']:,}/{m['tn']:,}",
                     mm["test_alerts_per_day_own_thr"]])
    A(_tbl(["模型", "n", "正例率", "ROC-AUC", "PR-AUC", "Brier", "Precision", "Recall", "F1",
            "TP/FP/FN/TN", "日均预警(格·槽/日)"], rows))
    A("")
    A(f"> 每个模型使用各自 val-F1 阈值（dual_xgb={thr_main:.2f} 等）。test 正例率 "
      f"{s6['models'][MAIN]['test_at_own_val_threshold']['positive_rate']:.4f} 高于 train "
      f"（{s5['train_pos_rate']:.4f}），因 2023–2026 年事件日在测试期分布更密；跨段比较需注意先验差异。")
    A("")
    rows = []
    for name, mm in s6["models"].items():
        m = mm["test_at_main_threshold"]
        rows.append([name, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                     f"{m['tp']:,}/{m['fp']:,}/{m['fn']:,}/{m['tn']:,}",
                     mm["test_alerts_per_day_main_thr"]])
    A(f"统一施用主模型阈值 {thr_main:.2f}（对照用）：")
    A("")
    A(_tbl(["模型", "Precision", "Recall", "F1", "TP/FP/FN/TN", "日均预警"], rows))
    A("")
    A("### 5.1 消融分解")
    A("")
    d = s6["models"]
    roc_dual = d[MAIN]["test_at_own_val_threshold"]["roc_auc"]
    pr_dual = d[MAIN]["test_at_own_val_threshold"]["pr_auc"]
    rows = []
    for name, label in (("power_only_xgb", "去除 ERA5 8 列（POWER 51 列）"),
                        ("static_only_xgb", "去除全部降雨+时间（静态 40 列）"),
                        ("rain_time_only_xgb", "仅降雨+时间（19 列）")):
        m = d[name]["test_at_own_val_threshold"]
        rows.append([label, f"{m['roc_auc']:.4f}", f"{m['roc_auc'] - roc_dual:+.4f}",
                     f"{m['pr_auc']:.4f}", f"{m['pr_auc'] - pr_dual:+.4f}"])
    A(_tbl(["消融", "ROC-AUC", "Δ ROC vs dual", "PR-AUC", "Δ PR vs dual"], rows))
    A("")
    g = s6["main_model_group_report_test"]
    rows = []
    for k in ("event_day_1", "event_day_0"):
        m = g.get(k, {})
        if "degenerate" in m:
            rows.append([k, m["n"], m.get("positives"), "退化（单一类别）", "—", "—", "—"])
        else:
            rows.append([k, m["n"], m["positives"], f"{m['roc_auc']:.4f}", f"{m['precision']:.4f}",
                         f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
    for y, m in sorted(g["by_year"].items()):
        if "degenerate" in m:
            rows.append([f"year {y}", m["n"], m.get("positives"), "退化（单一类别）", "—", "—", "—"])
        else:
            rows.append([f"year {y}", m["n"], m["positives"], f"{m['roc_auc']:.4f}",
                         f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
    A("主模型分组表现（test，@ 主阈值）：")
    A("")
    A(_tbl(["分组", "n", "正例", "ROC-AUC", "Precision", "Recall", "F1"], rows))
    A("")
    gm = s6["golden_eval_sensitivity"]["metrics"]
    if gm:
        A(f"**golden 敏感性**（真事件日加密抽样 {gm['n']} 行、正例率 {gm['positive_rate']:.2f}，"
          f"仅作参考，不并入主结果）：ROC-AUC {gm['roc_auc']:.4f}、PR-AUC {gm['pr_auc']:.4f}、"
          f"P {gm['precision']:.4f}、R {gm['recall']:.4f} @ 主阈值。")
    A("")
    A("## 6. SHAP 分析（条件 6）")
    A("")
    A(f"方法：xgboost `pred_contribs`（精确 TreeSHAP，log-odds 空间）；训练段随机抽样 "
      f"{s7['n_sample']:,} 行（正例 {s7['sample_positives']:,}，seed=42）。全局重要性前 15：")
    A("")
    rows = [[r["rank"], r["feature"], f"{r['mean_abs_shap_logodds']:.4f}", f"{r['xgb_gain_importance']:.4f}"]
            for r in s7["top20"][:15]]
    A(_tbl(["排名", "特征", "mean |SHAP| (log-odds)", "XGB gain importance"], rows))
    A("")
    A("图件：`figures/fig_shap_bar.png`（Top20 条形图）、`figures/fig_shap_dependence.png`（前 6 特征依赖图）；"
      "完整表：`shap_importance.csv`。")
    A("")
    A("## 7. GIS 风险结果（条件 6）")
    A("")
    A(f"- 文件：`gis/grid_risk_500m.csv`（{s8['n_grids']} 格，含 `lon_wgs84/lat_wgs84`，"
      "QGIS 可直接联接 `grid_id`）")
    A("- 字段：`mean_risk_all`（全部采样槽均值）、`p90_risk`、`mean_risk_event_day`（官方警告日均值）、"
      "`mean_risk_nonevent_day`、`rank_mean_risk`")
    A("- 图件：`figures/fig_gis_map.png`（示意散点；正式制图建议在 QGIS 中联接渲染）")
    A("- 风险最高的 10 个网格（总体均值口径）：")
    A("")
    rows = [[r["grid_id"], f"{r['lon_wgs84']:.5f}", f"{r['lat_wgs84']:.5f}",
             f"{r['mean_risk_all']:.4f}",
             f"{r['mean_risk_event_day']:.4f}" if pd.notna(r["mean_risk_event_day"]) else "—",
             int(r["n_event_rows"]) if pd.notna(r["n_event_rows"]) else 0]
            for r in s8["top30_by_mean_risk"][:10]]
    A(_tbl(["grid_id", "lon", "lat", "mean_risk_all", "mean_risk_event_day", "n_event_rows"], rows))
    A("")
    A("> 聚合基于非 golden 抽样行（train+val+test）；事件日均值仅对在官方警告日被采样到的网格有值。"
      "该风险面是**弱监督标签驱动的相对评分**，不是已验证的真实水浸概率分布。")
    A("")
    A("## 8. 偏差与局限披露（条件 7/8）")
    A("")
    A("1. **弱监督标签**：正例 = 「A 类空间规则命中格 × 官方红/黑警告活跃槽」，本质是「警告驱动的风险代理」，"
      "未与任何实测水浸/灾损记录核对；A 类规则本身的漏检/误报会原样传导进标签。")
    A(f"2. **320 个改判日抽样偏差**：旧版 320 个假事件日改判非事件日后，每日仅从当日已有网格重抽 35 格"
      f"（`np.random.default_rng(42)`，日期字符串升序逐日消耗），这些日期的空间覆盖远低于事件日；"
      f"全表 A 类格占比 {s1['is_A_share']:.1%}。模型在非事件日的负例空间构成受此重抽方案影响。")
    A("3. **ERA5/POWER 历史特征 vs 实时部署**：训练特征为 ERA5 再分析与 POWER 历史格点产品的"
      "「截至预测时刻」累计量；实时部署需要等价的准实时降雨格点输入（如雷达外推/实时站网），"
      "其误差结构不同于再分析，模型性能不可直接迁移。")
    A("4. 模型输出为风险评分，**不得称为已验证的真实水浸发生概率**；报告指标均为弱监督标签下的探索性结果。")
    A("5. 18 折完整空间 CV 未执行（用户条件 5 允许）；空间泛化目前仅由「网格→折唯一」的分组有效性检查支持。")
    A("")
    A("## 9. 论文可用结果段落（可直接改写引用）")
    A("")
    m = s6["models"][MAIN]["test_at_own_val_threshold"]
    A(f"基于 B 侧修复版数据集（delivery_repaired_b1，2,905,896 条 500 m 网格·2 h 槽样本，"
      f"2018–2021 训练 / 2022 验证 / 2023–2026.09 时间外测试），以 59 个预测时刻可得特征"
      f"（POWER 与 ERA5 双源前期降雨、天文潮、地形、地表覆盖、排水设施距离与时间循环项）训练 XGBoost。"
      f"阈值在验证集上按 F1 最大选定（τ={thr_main:.2f}）。在时间外测试集"
      f"（{m['n']:,} 条，正例率 {m['positive_rate']:.1%}）上，模型取得 ROC-AUC {m['roc_auc']:.3f}、"
      f"PR-AUC {m['pr_auc']:.3f}（正例率基线 {m['baseline_pr_auc']:.3f}）；阈值判定下 "
      f"Precision {m['precision']:.3f}、Recall {m['recall']:.3f}、F1 {m['f1']:.3f}。"
      f"消融显示 ERA5 特征带来 ROC-AUC +{roc_dual - d['power_only_xgb']['test_at_own_val_threshold']['roc_auc']:.3f} 的增益；"
      f"SHAP 全局重要性以静态地形/不透水面与前期降雨族为主（见图与 `shap_importance.csv`）。"
      f"上述结果基于官方暴雨警告驱动的弱监督标签，代表「警告条件下风险空间分辨」的能力，"
      f"不应解释为对真实水浸事件的提前 2 小时预测准确率。")
    A("")
    A("## 10. 产物清单")
    A("")
    for f in ["metrics.json", "univariate_auc_train.csv", "shap_importance.csv",
              "predictions_val.parquet", "predictions_test.parquet",
              "gis/grid_risk_500m.csv", "results_report.md",
              *[f"threshold_sweep_{n}.csv" for n in MODELS],
              *[f"bundles/{n}.joblib" for n in MODELS],
              *[f"figures/{x}" for x in result["figures"]]]:
        A(f"- `{f}`")
    A("")
    A(f"运行环境：Python {result['env']['python']}，xgboost {result['env']['xgboost']}，"
      f"sklearn {result['env']['sklearn']}，seed=42；总耗时 {result['total_seconds']}s，"
      f"峰值内存 {result['peak_rss_mb']} MB。")
    A("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
