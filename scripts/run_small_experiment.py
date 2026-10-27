"""小规模管线实验（任务书第四阶段任务 4.1 的验证版）。

目的**不是**产出论文指标，而是把 7 个管道环节逐个打通并留证据：

1. 数据读取与时间戳处理
2. 未来 2 小时目标对齐（含 prev 窗口锚定方向的重新实测）
3. 是否存在未来信息泄漏（单变量 AUC + 规则字段复现度）
4. 特征排除清单是否生效
5. train / val / test 划分是否正确
6. XGBoost / RandomForest / LogisticRegression 能否训练、预测、保存
7. 独立加载能否复现同一分数（子进程调用 scripts/predict_from_bundle.py）

用法：
    .venv/bin/python scripts/run_small_experiment.py --n-sample 300000
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
from sklearn.metrics import roc_auc_score

from floodrisk import models as M
from floodrisk import realdata as rd
from floodrisk.evaluate import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/delivery_v10/flood_risk_dataset_2h_500m.parquet"
THRESHOLD = 0.3  # 论文给定的参考阈值；不是治理成本验证过的预警阈值
KEY_COLS = ["grid_id", "date", "slot"]
RAIN_COLS = [
    "power_rain_slot_2h_mm",
    "power_rain_prev6h_mm",
    "power_rain_prev12h_mm",
    "power_rain_prev24h_mm",
    "power_rain_prev48h_mm",
]
# 锚定复测 + 对齐样例用；这些列**不进入**主模型
VERIF_COLS = RAIN_COLS + ["power_rain_prev3h_mm", "tide_pred_max_m"]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 特征侧检查
# ---------------------------------------------------------------------------
def registry_check(available: list[str]) -> dict:
    """排除清单 vs 主模型清单：不相交、且排除字段确实存在于交付数据中。"""
    feats, excluded = set(rd.FEATURES), set(rd.EXCLUDED)
    inter = sorted(feats & excluded)
    missing_excluded = sorted(c for c in excluded if c not in available)
    missing_feats = sorted(c for c in feats if c not in available)
    dup = sorted({c for c in rd.FEATURES if rd.FEATURES.count(c) > 1})
    return {
        "n_features": len(rd.FEATURES),
        "n_excluded": len(excluded),
        "feature_excluded_intersection": inter,
        "features_missing_in_parquet": missing_feats,
        "excluded_not_in_parquet": missing_excluded,
        "duplicated_features": dup,
        "ok": not inter and not missing_feats and not dup,
    }


def univariate_auc_scan(frame: pd.DataFrame, y_col: str, features: list[str]) -> list[dict]:
    """仅用训练集行做单变量 AUC；超过 0.97 视为疑似泄漏。"""

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
        r["suspicious"] = abs(r["auc"] - 0.5) >= 0.47  # 等价于 AUC>=0.97 或 <=0.03
    return rows


def anchoring_test(rows: pd.DataFrame) -> dict:
    """用相邻槽的 2h 雨量反推 prev6/12/24/48h 的锚点方向。

    槽 s 覆盖 [2s, 2s+2h)。若 prev6h 是 **槽开始** 锚定，则它等于
    前 3 个槽（s-3..s-1）的 2h 雨量之和；若 **槽结束** 锚定，则等于
    s-2..s（含本槽）。两个假设的 MAE 一比较，方向就唯一了。
    """
    d = rows.sort_values(["grid_id", "timestamp"]).copy()
    out: dict[str, dict] = {}
    for k in (6, 12, 24, 48):
        col = f"power_rain_prev{k}h_mm"
        if col not in d.columns:
            continue
        n = k // 2
        g = d.groupby("grid_id", observed=True)["power_rain_slot_2h_mm"]
        start = g.transform(lambda s: s.shift(1).rolling(n, min_periods=n).sum())
        end = g.transform(lambda s: s.rolling(n, min_periods=n).sum())
        ok = d[col].notna() & start.notna() & end.notna()
        out[str(k)] = {
            "n": int(ok.sum()),
            "mae_start_anchored": round(float((d.loc[ok, col] - start[ok]).abs().mean()), 6),
            "mae_end_anchored": round(float((d.loc[ok, col] - end[ok]).abs().mean()), 6),
            "mean_prev_mm": round(float(d.loc[ok, col].mean()), 4),
        }
    return out


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return binary_metrics(y, p, threshold=THRESHOLD, label=name)


def group_report(frame: pd.DataFrame, p: np.ndarray, y_col: str) -> dict:
    """整体 + 事件日/非事件日 + 逐年，分开报告（事件日高度集中，混合看会失真）。"""
    f = frame.assign(_p=p)
    out = {"overall": evaluate("overall", f[y_col].to_numpy(), p)}
    for flag in (1, 0):
        sub = f[f["is_event_day"].eq(flag)]
        if len(sub) and sub[y_col].nunique() == 2:
            out[f"event_day_{flag}"] = evaluate(f"event_day_{flag}", sub[y_col].to_numpy(), sub["_p"].to_numpy())
        elif len(sub):
            out[f"event_day_{flag}"] = {
                "label": f"event_day_{flag}",
                "n": int(len(sub)),
                "positives": int(sub[y_col].sum()),
                "degenerate": "单一类别，排序指标不可定义",
            }
    years = {}
    f = f.assign(_year=f["timestamp"].dt.year)
    for year, sub in f.groupby("_year"):
        if sub[y_col].nunique() == 2:
            years[str(int(year))] = evaluate(str(int(year)), sub[y_col].to_numpy(), sub["_p"].to_numpy())
        else:
            years[str(int(year))] = {
                "label": str(int(year)),
                "n": int(len(sub)),
                "positives": int(sub[y_col].sum()),
                "degenerate": "单一类别",
            }
    out["by_year"] = years
    return out


def compact(m: dict) -> str:
    keys = ("n", "positives", "roc_auc", "pr_auc", "precision", "recall", "f1")
    return " ".join(
        f"{k}={m.get(k)}" for k in keys
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="小规模管线实验")
    ap.add_argument("--n-sample", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) if args.outdir else ROOT / f"outputs/small_experiment_{stamp}"
    (outdir / "bundles").mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    result: dict = {
        "created_at": stamp,
        "data": {"path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA)},
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git": git_commit(),
        },
        "params": {"n_sample": args.n_sample, "seed": args.seed, "threshold": THRESHOLD},
    }

    import pyarrow
    import pyarrow.parquet as pq
    import sklearn
    import xgboost

    result["env"].update(
        {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pyarrow": pyarrow.__version__}
    )

    # ---- 1. 读键列（全表）--------------------------------------------------
    t = time.time()
    keys_cols = KEY_COLS + [
        "split",
        "cv_fold",
        "dc",
        "is_event_day",
        "is_A",
        "warn_level",
        "flooded_2h",
        "label_layer",
        "label_source",
        "confidence",
        "is_golden",
    ]
    df = rd.load_delivery(
        str(DATA),
        columns=keys_cols,
        categorical=["grid_id", "date", "label_layer", "label_source"],
    )
    result["step1_read"] = {
        "seconds": round(time.time() - t, 1),
        "rows": int(len(df)),
        "cols": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in ("grid_id", "date", "slot", "timestamp")},
        "timestamp_equals_date_plus_2h_slot": bool(
            (
                df["timestamp"]
                == pd.to_datetime(df["date"].astype(object).astype(str), format="%Y-%m-%d")
                + pd.to_timedelta(df["slot"].astype(int) * 2, unit="h")
            ).all()
        ),
        "timestamp_na": int(df["timestamp"].isna().sum()),
        "slot_range": [int(df["slot"].min()), int(df["slot"].max())],
        "date_range": [min(df["date"].cat.categories), max(df["date"].cat.categories)],
        "duplicated_grid_date_slot": int(df.duplicated(KEY_COLS).sum()),
        "peak_rss_mb": peak_rss_mb(),
    }
    print(f"[1] read {len(df):,} rows in {result['step1_read']['seconds']}s")

    # ---- 2. 目标对齐 -------------------------------------------------------
    df, target_report = rd.build_targets(df)
    result["step2_target"] = target_report
    result["step2_target"]["rule_determinism"] = {
        "flooded_2h_equals_is_A_and_warn_level_ge_2": round(
            float(
                (
                    df["flooded_2h"].astype(int)
                    == ((df["is_A"].astype(int) == 1) & (df["warn_level"].astype(int) >= 2)).astype(int)
                ).mean()
            ),
            6,
        ),
        "positives_total": int(df["flooded_2h"].sum()),
        "positives_explained_by_rule": int(
            (
                (df["flooded_2h"].astype(int) == 1)
                & ((df["is_A"].astype(int) == 1) & (df["warn_level"].astype(int) >= 2))
            ).sum()
        ),
    }
    print(f"[2] target rows={target_report['n_rows']:,} pos_rate={target_report['target_positive_rate']:.4f}")

    # ---- 3. 划分 -----------------------------------------------------------
    df, split_info = rd.assign_splits(df)
    partition = rd.check_partition(df)
    result["step3_split"] = {"info": split_info, "partition": partition}
    print(f"[3] split counts={split_info['counts']} partition_overlap={partition}")

    # ---- 4. 锚定方向复测 + 对齐样例（独立小样本、连续槽）-------------------
    date_cats = df["date"].cat.categories
    date_codes = df["date"].cat.codes.to_numpy()
    grid_cats = df["grid_id"].cat.categories
    grid_codes = df["grid_id"].cat.codes.to_numpy()

    ev_flag = np.zeros(len(date_cats), dtype=bool)
    ev_flag[np.unique(date_codes[df["is_event_day"].eq(1).to_numpy()])] = True
    # 找最长的连续"事件日"区间：非事件日每天只有 35 个网格（A 的抽样设计），
    # 混进来会让"逐槽滚动求和"在缺口处静默错位，所以验证窗口只用连续事件日。
    runs, start_i = [], None
    for i, flag in enumerate(ev_flag):
        if flag and start_i is None:
            start_i = i
        elif not flag and start_i is not None:
            runs.append((start_i, i))
            start_i = None
    if start_i is not None:
        runs.append((start_i, len(ev_flag)))
    if not runs:
        raise RuntimeError("数据中没有连续事件日，无法做锚定复测")
    run_lo, run_hi = max(runs, key=lambda r: r[1] - r[0])
    window_codes = list(range(run_lo, min(run_lo + 10, run_hi)))
    verif_dates = [date_cats[c] for c in window_codes]
    verif_mask = np.isin(date_codes, window_codes)

    present = np.unique(grid_codes[verif_mask])
    rng = np.random.default_rng(args.seed)
    chosen_grid_codes = rng.choice(present, size=min(60, len(present)), replace=False)
    grids = [str(grid_cats[c]) for c in chosen_grid_codes]
    vmask = np.isin(grid_codes, chosen_grid_codes) & verif_mask
    verif_keys = df.loc[vmask, KEY_COLS]

    verif = rd.load_delivery(
        str(DATA),
        columns=KEY_COLS + ["flooded_2h"] + VERIF_COLS,
        filter_keys=verif_keys,
        categorical=["grid_id", "date"],
    )
    # 只保留 12 槽齐全的网格，保证滚动窗口对应的是真实连续时间
    per_grid = verif.groupby(verif["grid_id"].astype(str)).size()
    full_grids = set(per_grid[per_grid == 12 * len(verif_dates)].index)
    verif = verif[verif["grid_id"].astype(str).isin(full_grids)]
    grids = [g for g in grids if g in full_grids]
    verif, _ = rd.build_targets(verif)
    result["step4_anchoring"] = {
        "verif_rows": int(len(verif)),
        "verif_grids_used": len(grids),
        "verif_dates": [verif_dates[0], verif_dates[-1]],
        "horizons": anchoring_test(verif),
        "interpretation": (
            "MAE(start-anchored) ≈ 0 且 MAE(end-anchored) 明显大于 0 "
            "→ prev 窗口截至槽开始，行的预测时刻就是槽开始，标签 flooded_2h 已是 (t, t+2h]"
        ),
    }

    # 对齐样例：2 个网格 × 前 3 天（含跨日 slot 11 → 次日 slot 0）
    demo_grids = grids[:2]
    demo_dates = verif_dates[:3]
    demo = verif[
        verif["grid_id"].astype(object).astype(str).isin(demo_grids)
        & verif["date"].astype(object).astype(str).isin(demo_dates)
    ]
    examples = rd.alignment_examples(demo, n_slots=len(demo))
    examples.to_csv(outdir / "alignment_examples.csv", index=False, encoding="utf-8-sig")
    result["step4_anchoring"]["example_rows"] = int(len(examples))
    print(f"[4] anchoring: {json.dumps(result['step4_anchoring']['horizons'], ensure_ascii=False)}")

    # ---- 5. 分层抽样 -------------------------------------------------------
    sample = rd.stratified_sample(df, args.n_sample, seed=args.seed)
    sample = sample.drop(columns=["year", "stratum"], errors="ignore")
    composition = (
        sample.groupby(["exp_split", "is_event_day"], observed=True)
        .agg(n=("slot", "size"), pos=("y_future_2h", "mean"))
        .reset_index()
    )
    result["step5_sample"] = {
        "n": int(len(sample)),
        "composition": composition.to_dict("records"),
        "years": {str(k): int(v) for k, v in sample["date_dt"].dt.year.value_counts().sort_index().items()},
        "n_grids": int(sample["grid_id"].nunique()),
        "label_rate_by_is_A": {
            str(k): {"n": int(v["size"]), "positive_rate": round(float(v["mean"]), 6)}
            for k, v in sample.groupby("is_A")["y_future_2h"].agg(["size", "mean"]).iterrows()
        },
        "note": "非事件日当天只有 35 个网格进入交付数据集（A 的抽样设计），全网格覆盖仅发生在事件日",
    }
    print(f"[5] sample {len(sample):,} rows, {sample['grid_id'].nunique()} grids")

    sample_keys = sample[KEY_COLS].copy()
    del df, verif, verif_keys, demo
    gc.collect()

    # ---- 6. 读入样本特征 ---------------------------------------------------
    t = time.time()
    feats = rd.load_delivery(
        str(DATA),
        columns=KEY_COLS + rd.FEATURES,
        filter_keys=sample_keys,
        categorical=["grid_id", "date"],
    )
    feats["grid_id"] = feats["grid_id"].astype(object).astype(str)
    feats["date"] = feats["date"].astype(object).astype(str)
    feats = feats.drop(columns=["timestamp"], errors="ignore")  # 与键表重复，避免 _x/_y
    sample["grid_id"] = sample["grid_id"].astype(object).astype(str)
    sample["date"] = sample["date"].astype(object).astype(str)
    data = sample.merge(feats, on=KEY_COLS, how="left", validate="one_to_one")
    del feats
    gc.collect()

    result["step6_features"] = {
        "read_seconds": round(time.time() - t, 1),
        "rows": int(len(data)),
        "feature_na_rate": {
            c: round(float(data[c].isna().mean()), 4) for c in rd.FEATURES if data[c].isna().any()
        },
        "registry": registry_check(pq.read_schema(str(DATA)).names),
        "peak_rss_mb": peak_rss_mb(),
    }
    print(f"[6] features read in {result['step6_features']['read_seconds']}s, ok={result['step6_features']['registry']['ok']}")

    tr = data[data["exp_split"].eq("train")]
    va = data[data["exp_split"].eq("val")]
    te = data[data["exp_split"].eq("test")]
    print(f"    train={len(tr):,} val={len(va):,} test={len(te):,}")

    # ---- 7. 泄漏体检 -------------------------------------------------------
    scan = univariate_auc_scan(tr, "y_future_2h", rd.FEATURES)
    result["step7_leakage"] = {
        "univariate_auc_train": scan,
        "suspicious_features": [r["feature"] for r in scan if r["suspicious"]],
        "top10": scan[:10],
    }
    print(f"[7] suspicious AUC(>=0.97): {result['step7_leakage']['suspicious_features']}")

    # ---- 8. 训练三模型 -----------------------------------------------------
    X_tr = tr[rd.FEATURES]
    y_tr = tr["y_future_2h"].to_numpy(dtype=int)
    X_va, y_va = va[rd.FEATURES], va["y_future_2h"].to_numpy(dtype=int)
    X_te, y_te = te[rd.FEATURES], te["y_future_2h"].to_numpy(dtype=int)

    models_info: dict = {}
    preds: dict[str, np.ndarray] = {}
    for backend in ("xgb", "rf", "lr"):
        t = time.time()
        model = M.build_model(backend, seed=args.seed)
        model.fit(X_tr, y_tr)
        seconds = round(time.time() - t, 1)
        p_va = M.predict_proba(model, X_va)
        p_te = M.predict_proba(model, X_te)
        preds[backend] = p_te
        bundle = M.Bundle(
            backend=backend,
            model=model,
            features=list(rd.FEATURES),
            meta={
                "version": f"small_exp_{stamp}_{backend}",
                "n_train": int(len(X_tr)),
                "seed": args.seed,
                "target": "y_future_2h",
                "threshold_reference": THRESHOLD,
                "data_sha256": result["data"]["sha256"],
                "exploratory": True,
            },
        )
        M.save_bundle(outdir / "bundles" / f"{backend}.joblib", bundle)
        models_info[backend] = {
            "fit_seconds": seconds,
            "val": evaluate(f"{backend}_val", y_va, p_va),
            "test": group_report(te, p_te, "y_future_2h"),
        }
        if backend == "xgb":
            imp = pd.Series(model.feature_importances_, index=rd.FEATURES).sort_values(ascending=False)
            models_info["xgb"]["importance_top15"] = {
                str(k): round(float(v), 6) for k, v in imp.head(15).items()
            }
        print(f"[8] {backend:3s} fit {seconds}s | test {compact(models_info[backend]['test']['overall'])}")

    result["step8_models"] = models_info

    # ---- 9. 对照：规则基线 + 强泄漏版 XGB（仅作证伪，不当结果）-------------
    rule_tr = rd.rule_baseline(tr)
    rule_te = rd.rule_baseline(te)
    result["step9_controls"] = {
        "rule_baseline_test": evaluate("rule_baseline_test", y_te, rule_te.astype(float)),
        "rule_baseline_recall_note": "规则即标签生成器，召回接近 1 是定义使然，不是预警能力",
        "rule_baseline_train_positive_rate": round(float(rule_tr.mean()), 6),
    }

    leak_cols = list(rd.FEATURES) + ["is_A", "warn_level", "warn_level_day", "is_event_day"]
    leak_cols = [c for c in leak_cols if c in data.columns]
    if len(leak_cols) > len(rd.FEATURES):
        t = time.time()
        leak_model = M.build_model("xgb", seed=args.seed)
        leak_model.fit(tr[leak_cols], y_tr)
        p_leak = M.predict_proba(leak_model, te[leak_cols])
        result["step9_controls"]["leaky_reference_xgb"] = {
            "columns_added": [c for c in leak_cols if c not in rd.FEATURES],
            "fit_seconds": round(time.time() - t, 1),
            "test": evaluate("leaky_reference", y_te, p_leak),
        }
        print(
            f"[9] leaky-reference test {compact(result['step9_controls']['leaky_reference_xgb']['test'])} "
            "（对标主模型的差距即为排除清单的作用）"
        )

    # ---- 9b. 特征族分解：风险分数到底来自"空间记忆"还是"降雨驱动" -----------
    rain_cols = [c for c in rd.FEATURES if c.startswith("power_rain")]
    time_cols = ["month", "hour_sin", "hour_cos"]
    static_cols = [c for c in rd.FEATURES if c not in rain_cols and c not in time_cols]
    decomposition = {}
    for name, cols in (("static_only", static_cols), ("rain_time_only", rain_cols + time_cols)):
        t = time.time()
        mdl = M.build_model("xgb", seed=args.seed)
        mdl.fit(tr[cols], y_tr)
        decomposition[name] = {
            "n_features": len(cols),
            "fit_seconds": round(time.time() - t, 1),
            "test": evaluate(name, y_te, M.predict_proba(mdl, te[cols])),
        }
        print(f"[9b] {name:14s} test {compact(decomposition[name]['test'])}")
    result["step9_controls"]["feature_family_decomposition"] = decomposition

    # ---- 10. 敏感性：槽结束锚定目标（前移 1 槽）----------------------------
    shift_ok_tr = tr["y_future_2h_shift"].notna()
    shift_ok_te = te["y_future_2h_shift"].notna()
    shift_model = M.build_model("xgb", seed=args.seed)
    t = time.time()
    shift_model.fit(X_tr[shift_ok_tr], tr.loc[shift_ok_tr, "y_future_2h_shift"].to_numpy(dtype=int))
    p_shift = M.predict_proba(shift_model, X_te[shift_ok_te])
    shift_bundle = M.Bundle(
        backend="xgb",
        model=shift_model,
        features=list(rd.FEATURES),
        meta={
            "version": f"small_exp_{stamp}_xgb_shift",
            "target": "y_future_2h_shift（槽结束锚定，敏感性对照）",
            "n_train": int(shift_ok_tr.sum()),
            "seed": args.seed,
            "exploratory": True,
        },
    )
    M.save_bundle(outdir / "bundles" / "xgb_shift.joblib", shift_bundle)
    result["step10_shift_sensitivity"] = {
        "fit_seconds": round(time.time() - t, 1),
        "n_train": int(shift_ok_tr.sum()),
        "n_test": int(shift_ok_te.sum()),
        "test": evaluate(
            "xgb_shift_test", te.loc[shift_ok_te, "y_future_2h_shift"].to_numpy(dtype=int), p_shift
        ),
        "note": "只作方向性对照：若主目标无需位移，这一版不应显著更优",
    }
    print(f"[10] shift sensitivity test {compact(result['step10_shift_sensitivity']['test'])}")

    # ---- 11. 独立加载复现 --------------------------------------------------
    b = M.load_bundle(outdir / "bundles" / "xgb.joblib")
    p_reload = M.predict_with_bundle(b, X_te.head(5000))
    same = float(np.max(np.abs(p_reload - preds["xgb"][:5000])))

    probe = te.head(5000)[KEY_COLS + ["timestamp"] + rd.FEATURES].copy()
    probe.to_parquet(outdir / "probe_rows.parquet", index=False)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/predict_from_bundle.py"),
        "--bundle",
        str(outdir / "bundles/xgb.joblib"),
        "--input",
        str(outdir / "probe_rows.parquet"),
        "--out",
        str(outdir / "probe_preds.csv"),
    ]
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    cli_out = proc.stdout.strip()
    cli_err = proc.stderr.strip()[-800:]
    cli_csv = outdir / "probe_preds.csv"
    cli_max_diff = None
    if cli_csv.exists():
        got = pd.read_csv(cli_csv)["risk_score"].to_numpy()
        cli_max_diff = float(np.max(np.abs(got - preds["xgb"][:5000])))
    result["step11_standalone"] = {
        "in_process_max_abs_diff": same,
        "cli_returncode": proc.returncode,
        "cli_seconds": round(time.time() - t, 1),
        "cli_stdout": cli_out,
        "cli_stderr_tail": cli_err,
        "cli_max_abs_diff": cli_max_diff,
        "cli_ok": proc.returncode == 0 and cli_max_diff is not None and cli_max_diff < 1e-9,
    }
    print(f"[11] standalone cli ok={result['step11_standalone']['cli_ok']} diff={cli_max_diff}")

    # ---- 12. 落盘 -----------------------------------------------------------
    write_json(outdir / "metrics.json", result)
    write_json(
        outdir / "feature_registry.json",
        {
            "features": rd.FEATURES,
            "groups": rd.FEATURE_GROUPS,
            "rule_columns_not_in_model": rd.RULE_COLUMNS,
            "excluded_with_reasons": rd.EXCLUDED,
        },
    )
    sample.groupby(["exp_split", "is_event_day"], observed=True).size().to_csv(
        outdir / "sample_composition.csv"
    )
    pd.DataFrame(scan).to_csv(outdir / "univariate_auc.csv", index=False)

    result["total_seconds"] = round(time.time() - t_start, 1)
    result["peak_rss_mb"] = peak_rss_mb()
    write_json(outdir / "metrics.json", result)
    print(f"[done] {result['total_seconds']}s, peak RSS {result['peak_rss_mb']} MB -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
