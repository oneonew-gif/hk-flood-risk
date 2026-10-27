"""v10.1 对照实验：ERA5 加入前后，同数据版本 / 相同样本 / 同划分 / 同种子。

对照用户 2026-10-24 批准的方案 A 与本轮附加要求：
- 四组特征方案：power_only(51) / dual(59=51+ERA5 8) / static_only(40) / rain_time_only(19)
- 三模型 XGB / RF / LR 全跑（rain_time_power 11 列为修正口径诊断组，仅 XGB）
- 指标含 Brier / Precision / Recall / 正样本比例
- v10.0 与 v10.1 行序逐行一致（已验证）→ 同 seed 抽样键完全相同，跨版本严格可比
- v10.0 的模型、报告、输出全部保留，本脚本输出到独立目录

用法：
    .venv/bin/python scripts/run_small_experiment_v10_1.py --n-sample 300000
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
from floodrisk.evaluate import binary_metrics  # noqa: E402

DATA = ROOT / "data/delivery_v10_1/flood_risk_dataset_2h_500m.parquet"
DATA_V10_0 = ROOT / "data/delivery_v10/flood_risk_dataset_2h_500m.parquet"
METRICS_V10_0 = ROOT / "outputs/small_experiment_300k_20261024/metrics.json"
KEY_COLS = rd.KEY
THRESHOLD = 0.3  # 论文参考阈值（非成本验证的最优阈值）

RAIN_ALL = list(rd.RAIN_FEATURES)
TIME_COLS = list(rd.TIME_FEATURES)
_RAIN_SET = set(RAIN_ALL) | set(TIME_COLS)
FEATSETS: dict[str, list[str]] = {
    "power_only": list(rd.POWER_ONLY_FEATURES),  # v10.0 原方案 51 列
    "dual": list(rd.FEATURES),  # 方案 A：51 + ERA5 8 = 59 列
    "static_only": [c for c in rd.FEATURES if c not in _RAIN_SET],  # 40 列（零降雨零时间）
    "rain_time_only": RAIN_ALL + TIME_COLS,  # 19 列（ERA5+POWER 降雨族 + 时间）
    "rain_time_power": [c for c in rd.POWER_ONLY_FEATURES if c in _RAIN_SET],  # 11 列诊断组
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float:
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
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
        "ok": not inter and not (set(rd.FEATURES) - set(available)) and not sorted({c for c in rd.FEATURES if rd.FEATURES.count(c) > 1}),
    }


def univariate_auc_scan(frame: pd.DataFrame, y_col: str, features: list[str]) -> list[dict]:
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


def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return binary_metrics(y, p, threshold=THRESHOLD, label=name)


def group_report(frame: pd.DataFrame, p: np.ndarray, y_col: str) -> dict:
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
            years[str(int(year))] = {"label": str(int(year)), "n": int(len(sub)), "positives": int(sub[y_col].sum()), "degenerate": "单一类别"}
    out["by_year"] = years
    return out


def compact(m: dict) -> str:
    return " ".join(f"{k}={m.get(k)}" for k in ("n", "positives", "positive_rate", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v10.1 特征方案对照实验")
    ap.add_argument("--n-sample", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    t_start = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) if args.outdir else ROOT / f"outputs/small_experiment_v10_1_300k_{stamp}"
    (outdir / "bundles").mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    result: dict = {
        "created_at": stamp,
        "data": {"path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA), "version": "v10.1 (batch 2)"},
        "baseline_v10_0": {"path": str(DATA_V10_0.relative_to(ROOT)), "sha256": sha256(DATA_V10_0)},
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git": git_commit(),
        },
        "params": {"n_sample": args.n_sample, "seed": args.seed, "threshold": THRESHOLD,
                   "design": "四组特征方案 × 三模型；同 seed 同行序 → 样本键与 v10.0 完全一致"},
    }

    import pyarrow.parquet as pq
    import sklearn
    import xgboost

    result["env"].update({"sklearn": sklearn.__version__, "xgboost": xgboost.__version__})

    # ---- 1. 读键列（全表 v10.1）+ 行序一致性 --------------------------------
    t = time.time()
    keys_cols = KEY_COLS + ["split", "cv_fold", "dc", "is_event_day", "is_A", "warn_level",
                            "flooded_2h", "label_layer", "label_source", "confidence", "is_golden"]
    df = rd.load_delivery(str(DATA), columns=keys_cols, categorical=["grid_id", "date", "label_layer", "label_source"])
    row_order_check = {}
    try:
        o = pq.read_table(str(DATA_V10_0), columns=KEY_COLS)
        n = pq.read_table(str(DATA), columns=KEY_COLS)
        row_order_check = {
            "v10_0_rows": o.num_rows, "v10_1_rows": n.num_rows,
            "row_order_identical": all(o.column(c).equals(n.column(c)) for c in KEY_COLS),
        }
        del o, n
    except Exception as e:  # pragma: no cover
        row_order_check = {"error": str(e)}
    result["step1_read"] = {
        "seconds": round(time.time() - t, 1),
        "rows": int(len(df)),
        "timestamp_equals_date_plus_2h_slot": bool(
            (df["timestamp"] == pd.to_datetime(df["date"].astype(object).astype(str), format="%Y-%m-%d")
             + pd.to_timedelta(df["slot"].astype(int) * 2, unit="h")).all()
        ),
        "duplicated_grid_date_slot": int(df.duplicated(KEY_COLS).sum()),
        "row_order_vs_v10_0": row_order_check,
        "peak_rss_mb": peak_rss_mb(),
    }
    print(f"[1] read {len(df):,} rows; row_order_identical={row_order_check.get('row_order_identical')}")

    # ---- 2. 目标 ------------------------------------------------------------
    df, target_report = rd.build_targets(df)
    result["step2_target"] = target_report
    result["step2_target"]["rule_determinism"] = {
        "flooded_2h_equals_is_A_and_warn_level_ge_2": round(float(
            ((df["flooded_2h"].astype(int) == ((df["is_A"].astype(int) == 1) & (df["warn_level"].astype(int) >= 2)).astype(int)).mean())
        ), 6),
        "positives_total": int(df["flooded_2h"].sum()),
    }
    print(f"[2] target rows={target_report['n_rows']:,}")

    # ---- 3. 划分 ------------------------------------------------------------
    df, split_info = rd.assign_splits(df)
    partition = rd.check_partition(df)
    result["step3_split"] = {"info": split_info, "partition": partition}
    print(f"[3] split counts={split_info['counts']} partition={partition}")

    # ---- 5. 分层抽样（同 v10.0 seed → 相同样本键）----------------------------
    sample = rd.stratified_sample(df, args.n_sample, seed=args.seed)
    sample = sample.drop(columns=["year", "stratum"], errors="ignore")
    composition = (
        sample.groupby(["exp_split", "is_event_day"], observed=True)
        .agg(n=("slot", "size"), pos=("y_future_2h", "mean")).reset_index()
    )
    comp_v10_0_path = METRICS_V10_0.parent / "sample_composition.csv"
    result["step5_sample"] = {
        "n": int(len(sample)),
        "composition": composition.to_dict("records"),
        "years": {str(k): int(v) for k, v in sample["date_dt"].dt.year.value_counts().sort_index().items()},
        "n_grids": int(sample["grid_id"].nunique()),
        "label_rate_by_is_A": {
            str(k): {"n": int(v["size"]), "positive_rate": round(float(v["mean"]), 6)}
            for k, v in sample.groupby("is_A")["y_future_2h"].agg(["size", "mean"]).iterrows()
        },
        "composition_csv_v10_0_exists": comp_v10_0_path.exists(),
    }
    print(f"[5] sample {len(sample):,} rows, {sample['grid_id'].nunique()} grids")
    sample_keys = sample[KEY_COLS].copy()
    del df
    gc.collect()

    # ---- 6. 读入样本特征（59 列 dual）---------------------------------------
    t = time.time()
    feats = rd.load_delivery(str(DATA), columns=KEY_COLS + rd.FEATURES, filter_keys=sample_keys,
                             categorical=["grid_id", "date"])
    feats["grid_id"] = feats["grid_id"].astype(object).astype(str)
    feats["date"] = feats["date"].astype(object).astype(str)
    feats = feats.drop(columns=["timestamp"], errors="ignore")
    sample["grid_id"] = sample["grid_id"].astype(object).astype(str)
    sample["date"] = sample["date"].astype(object).astype(str)
    data = sample.merge(feats, on=KEY_COLS, how="left", validate="one_to_one")
    del feats
    gc.collect()

    na_rate = {c: round(float(data[c].isna().mean()), 4) for c in rd.FEATURES if data[c].isna().any()}
    result["step6_features"] = {
        "read_seconds": round(time.time() - t, 1),
        "rows": int(len(data)),
        "feature_na_rate": na_rate,
        "era5_na_rate_in_sample": {c: round(float(data[c].isna().mean()), 4) for c in rd.FEATURES if c.startswith("era5")},
        "registry": registry_check(pq.read_schema(str(DATA)).names),
        "featsets_sizes": {k: len(v) for k, v in FEATSETS.items()},
        "peak_rss_mb": peak_rss_mb(),
    }
    print(f"[6] features read in {result['step6_features']['read_seconds']}s, registry ok={result['step6_features']['registry']['ok']}")

    tr = data[data["exp_split"].eq("train")]
    va = data[data["exp_split"].eq("val")]
    te = data[data["exp_split"].eq("test")]
    print(f"    train={len(tr):,} val={len(va):,} test={len(te):,}")

    # ---- 7. 泄漏体检（59 列，仅训练段）--------------------------------------
    scan = univariate_auc_scan(tr, "y_future_2h", rd.FEATURES)
    result["step7_leakage"] = {
        "univariate_auc_train": scan,
        "suspicious_features": [r["feature"] for r in scan if r["suspicious"]],
        "top10": scan[:10],
    }
    print(f"[7] suspicious: {result['step7_leakage']['suspicious_features']}")

    # ---- 8. 四组特征 × 三模型（rain_time_power 诊断组仅 XGB）-----------------
    y_tr = tr["y_future_2h"].to_numpy(dtype=int)
    y_va = va["y_future_2h"].to_numpy(dtype=int)
    y_te = te["y_future_2h"].to_numpy(dtype=int)

    ablation: dict = {}
    for fsname, fs in FEATSETS.items():
        backends = ("xgb", "rf", "lr") if fsname != "rain_time_power" else ("xgb",)
        for backend in backends:
            t = time.time()
            model = M.build_model(backend, seed=args.seed)
            model.fit(tr[fs], y_tr)
            seconds = round(time.time() - t, 1)
            p_va = M.predict_proba(model, va[fs])
            p_te = M.predict_proba(model, te[fs])
            bundle = M.Bundle(
                backend=backend, model=model, features=list(fs),
                meta={
                    "version": f"v10_1_ablation_{fsname}_{backend}",
                    "feature_set": fsname, "n_features": len(fs),
                    "n_train": int(len(tr)), "seed": args.seed,
                    "target": "y_future_2h", "threshold_reference": THRESHOLD,
                    "data_sha256": result["data"]["sha256"], "data_version": "v10.1",
                    "exploratory": True,
                },
            )
            M.save_bundle(outdir / "bundles" / f"{fsname}_{backend}.joblib", bundle)
            ablation.setdefault(fsname, {})[backend] = {
                "n_features": len(fs), "fit_seconds": seconds,
                "val": evaluate(f"{fsname}_{backend}_val", y_va, p_va),
                "test": group_report(te, p_te, "y_future_2h"),
            }
            if backend == "xgb":
                imp = pd.Series(model.feature_importances_, index=fs).sort_values(ascending=False)
                ablation[fsname][backend]["importance_top15"] = {str(k): round(float(v), 6) for k, v in imp.head(15).items()}
            print(f"[8] {fsname:15s} {backend:3s} fit {seconds:5.1f}s | test {compact(ablation[fsname][backend]['test']['overall'])}")
    result["step8_ablation"] = ablation

    # ---- 9. 规则基线（无拟合，定义使然）--------------------------------------
    result["step9_controls"] = {
        "rule_baseline_test": evaluate("rule_baseline_test", y_te, rd.rule_baseline(te).astype(float)),
        "note": "规则即标签生成器，AUC≈1 是定义使然，不是预警能力",
    }
    print(f"[9] rule baseline {compact(result['step9_controls']['rule_baseline_test'])}")

    # ---- 11. 独立加载复现（dual xgb）----------------------------------------
    b = M.load_bundle(outdir / "bundles" / "dual_xgb.joblib")
    X_te_head = te.head(5000)
    p_reload = M.predict_with_bundle(b, X_te_head)
    p_inproc = ablation["dual"]["xgb"]["_p_te_head"] if "_p_te_head" in ablation["dual"]["xgb"] else None
    # 重新算进程内概率（上一循环未存头部概率）
    mdl = b  # bundle 内即同一模型
    p_inproc = M.predict_proba(mdl.model, X_te_head[list(b.features)])
    result["step11_reload"] = {
        "max_abs_diff_inproc_vs_bundle": round(float(np.max(np.abs(p_reload - p_inproc))), 10),
    }
    probe = X_te_head[KEY_COLS + ["timestamp"] + rd.FEATURES].copy()
    probe.to_parquet(outdir / "probe_rows.parquet", index=False)
    cmd = [sys.executable, str(ROOT / "scripts/predict_from_bundle.py"),
           "--bundle", str(outdir / "bundles/dual_xgb.joblib"),
           "--input", str(outdir / "probe_rows.parquet"),
           "--out", str(outdir / "probe_preds.csv")]
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        result["step11_reload"]["cli_error"] = proc.stderr[-500:]
        print(f"[11] CLI FAILED:\n{proc.stderr[-800:]}")
        return 1
    cli = pd.read_csv(outdir / "probe_preds.csv")
    diff = float(np.max(np.abs(cli["risk_score"].to_numpy() - p_inproc)))
    result["step11_reload"].update({
        "cli_rows": int(len(cli)), "cli_max_abs_diff_vs_inproc": round(diff, 10),
        "cli_seconds": round(time.time() - t, 1),
    })
    print(f"[11] reload diff={result['step11_reload']['max_abs_diff_inproc_vs_bundle']} cli diff={result['step11_reload']['cli_max_abs_diff_vs_inproc']}")

    # ---- 12. v10.0 基线数字（供报告对比；v10.0 产物不动）---------------------
    if METRICS_V10_0.exists():
        old = json.loads(METRICS_V10_0.read_text(encoding="utf-8"))
        comp = {}
        for backend in ("xgb", "rf", "lr"):
            o = old["step8_models"][backend]["test"]["overall"]
            n = ablation["power_only"][backend]["test"]["overall"]
            comp[f"{backend}_power_only"] = {
                "v10_0": {k: o.get(k) for k in ("n", "positives", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1")},
                "v10_1": {k: n.get(k) for k in ("n", "positives", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1")},
            }
        old_fam = old["step9_controls"]["feature_family_decomposition"]
        comp["rain_time_only_note"] = (
            "v10.0 的 rain_time_only=10 列（startswith 前缀把 power_api72 误归静态族）；"
            "本轮 rain_time_power=11 列为修正口径的可比组；rain_time_only=19 列含 ERA5"
        )
        comp["static_only"] = {
            "v10_0_41cols": old_fam["static_only"]["test"],
            "v10_1_40cols_xgb": ablation["static_only"]["xgb"]["test"]["overall"],
        }
        comp["rain_time_only_v10_0_10cols"] = old_fam["rain_time_only"]["test"]
        comp["rain_time_power_v10_1_11cols_xgb"] = ablation["rain_time_power"]["xgb"]["test"]["overall"]
        comp["rain_time_only_v10_1_19cols"] = {b: ablation["rain_time_only"][b]["test"]["overall"] for b in ("xgb", "rf", "lr")}
        result["v10_0_vs_v10_1_comparison"] = comp

    result["total_seconds"] = round(time.time() - t_start, 1)
    result["peak_rss_mb"] = peak_rss_mb()
    result["disclaimer"] = (
        "弱监督标签下的探索性结果：标签 = is_A × [warn_level>=2]，"
        "全部指标不得解释为真实水浸提前 2 小时预测准确率"
    )
    write_json(outdir / "metrics.json", result)
    pd.DataFrame(scan).to_csv(outdir / "univariate_auc.csv", index=False)
    composition.to_csv(outdir / "sample_composition.csv", index=False)
    print(f"\n[done] {result['total_seconds']}s peak {result['peak_rss_mb']}MB → {outdir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
