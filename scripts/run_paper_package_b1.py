"""论文成果整理：最终模型对比表 + 六类图件（用户 2026-10-26 批准，只读整理，不训练）。

任务 → 实现
1. 最终模型对比表：6 模型（dual_xgb / dual_lr / power_only_xgb / static_only_xgb /
   rain_time_only_xgb / no_tide_xgb）test 指标，数值直接取自正式训练与补充实验
   的 metrics.json（不重算指标），预警数/日按 (tp+fp)/1369 复核。
2. 图件（全部由既有预测/CSV 生成，不训练）：
   - fig_paper_roc_pr_test.png：6 模型 test ROC + PR 曲线
   - fig_paper_rain_tide_heatmap.png：雨×潮 3 口径（全部/沿海/内陆）×
     实际正例率 / 平均评分 热力图
   - fig_paper_grid_risk_test_vs_full.png：测试期(2023–2026.09) vs 全期
     (2018–2026.09，含训练期，GIS mean_risk_all) 500 米网格风险地图并排，
     口径在标题中明确区分
   - fig_paper_rank_stability.png：dual_xgb vs no_tide_xgb 逐格排名散点 +
     Top20 重合标注（排名稳定性）
   - fig_paper_shap_top20.png：SHAP top20 条形图（来自正式训练 shap_importance.csv）
   - 另将正式训练 6 张原图复制到 package/formal_originals/ 备查
3. GIS 口径区分：测试期网格风险表 grid_risk_test_period_500m.csv（QGIS 可联接）
   与全期 GIS 表并存，论文优先使用测试期。
4. 表格：final_model_comparison.csv、grid_risk_test_period_500m.csv、
   雨潮复合表与阈值敏感性表复制汇集。

硬约束：只读 data/，不训练，不改原始数据与旧报告；不生成任何社区级排名。
全部产物写入独立目录 outputs/paper_package_b1_<ts>/。

用法：
    .venv/bin/python scripts/run_paper_package_b1.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet"
MANIFEST = ROOT / "data/delivery_repaired_b1/manifest_repair_b1.json"
FORMAL_DIR = ROOT / "outputs/formal_training_repaired_b1_20261026-020814"
SUPP_DIR = ROOT / "outputs/supplement_tide_ablation_b1_20261026-024612"
PRED_FORMAL = FORMAL_DIR / "predictions_test.parquet"
PRED_NOTIDE = SUPP_DIR / "predictions_test_no_tide.parquet"
GIS_CSV = FORMAL_DIR / "gis/grid_risk_500m.csv"
SHAP_CSV = FORMAL_DIR / "shap_importance.csv"
N_TEST_DATES = 1369

MODEL_ORDER = ["dual_xgb", "dual_lr", "power_only_xgb", "static_only_xgb",
               "rain_time_only_xgb", "no_tide_xgb"]
MODEL_LABEL = {
    "dual_xgb": "XGBoost（双源特征，主模型）",
    "dual_lr": "逻辑回归（双源特征，基线）",
    "power_only_xgb": "XGBoost（仅 POWER 降雨）",
    "static_only_xgb": "XGBoost（仅静态地形）",
    "rain_time_only_xgb": "XGBoost（仅降雨+时间）",
    "no_tide_xgb": "XGBoost（去潮位消融）",
}
MODEL_COLOR = {
    "dual_xgb": "#1f77b4", "dual_lr": "#d62728", "power_only_xgb": "#2ca02c",
    "static_only_xgb": "#9467bd", "rain_time_only_xgb": "#7f7f7f",
    "no_tide_xgb": "#ff7f0e",
}

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# 任务 1：最终模型对比表（数值取自既有 metrics.json，不重算）
# ---------------------------------------------------------------------------
def build_comparison_table(formal: dict, supp: dict, bundles_dir: Path) -> pd.DataFrame:
    fm = formal["step6_test_final"]["models"]
    thr_frozen = formal["step6_test_final"]["thresholds_frozen_on_val"]
    rows = []
    for m in MODEL_ORDER:
        if m == "no_tide_xgb":
            s = supp["step3_ablation"]["no_tide_test_own"]
            thr = supp["step3_ablation"]["no_tide_val_threshold"]
            src = "supplement_tide_ablation_b1 (no_tide_test_own)"
        else:
            s = fm[m]["test_at_own_val_threshold"]
            thr = thr_frozen[m]
            src = "formal_training_repaired_b1 (test_at_own_val_threshold)"
        rows.append({
            "model": m,
            "model_label": MODEL_LABEL[m],
            "n_features": None,  # 稍后由 bundle 补
            "threshold_val_frozen": thr,
            "test_roc_auc": s["roc_auc"],
            "test_pr_auc": s["pr_auc"],
            "test_brier": s["brier"],
            "test_precision": s["precision"],
            "test_recall": s["recall"],
            "test_f1": s["f1"],
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "alerts_per_day": round((s["tp"] + s["fp"]) / N_TEST_DATES, 2),
            "positives": s["positives"], "n": s["n"],
            "source": src,
        })
    df = pd.DataFrame(rows)
    # 特征数：从 bundle 读取（只读）
    for i, m in enumerate(df["model"]):
        bpath = bundles_dir / f"{m}.joblib"
        if m == "no_tide_xgb":
            bpath = SUPP_DIR / "bundles" / "no_tide_xgb.joblib"
        try:
            from floodrisk.models import load_bundle
            df.loc[i, "n_features"] = len(load_bundle(bpath).features)
        except Exception as e:  # noqa: BLE001
            df.loc[i, "n_features"] = f"读取失败({e})"
    # 相对主模型的差值
    base = df.loc[df["model"] == "dual_xgb"].iloc[0]
    df["delta_roc_vs_dual"] = (df["test_roc_auc"] - base["test_roc_auc"]).round(6)
    df["delta_pr_vs_dual"] = (df["test_pr_auc"] - base["test_pr_auc"]).round(6)
    return df


# ---------------------------------------------------------------------------
# 任务 2a：ROC / PR 曲线（6 模型）
# ---------------------------------------------------------------------------
def fig_roc_pr(pf: pd.DataFrame, pn: pd.DataFrame, figdir: Path) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = pf["y_future_2h"].to_numpy(dtype=int)
    series = {m: pf[f"p_{m}"].to_numpy(dtype=float)
              for m in MODEL_ORDER if m != "no_tide_xgb"}
    series["no_tide_xgb"] = pn["p_no_tide_xgb"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    stats = {}
    for m in MODEL_ORDER:
        p = series[m]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        from sklearn.metrics import average_precision_score, roc_auc_score
        roc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        stats[m] = {"roc_auc": round(float(roc), 6), "pr_auc": round(float(ap), 6)}
        lab = f"{MODEL_LABEL[m]}（ROC {roc:.3f} / PR {ap:.3f}）"
        axes[0].plot(fpr, tpr, color=MODEL_COLOR[m], lw=1.8, label=lab)
        axes[1].plot(rec, prec, color=MODEL_COLOR[m], lw=1.8, label=lab)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8, alpha=0.6,
                    label=f"正例率基线 {y.mean():.3f}")
    axes[0].set(xlabel="假正例率", ylabel="真正例率", title="(a) 测试期 ROC 曲线（2023–2026.09）",
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="召回率", ylabel="精确率", title="(b) 测试期 PR 曲线（2023–2026.09）",
                xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend(fontsize=7.2, loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(alpha=0.25)
    fig.suptitle("弱监督标签（官方暴雨警告×空间规则）下的探索性结果；非真实水浸提前 2 小时预测准确率",
                 fontsize=8.5, y=0.02, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(figdir / "fig_paper_roc_pr_test.png", dpi=150)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 任务 2b：雨潮分组热力图（3 口径 × 正例率 / 平均评分）
# ---------------------------------------------------------------------------
def fig_rain_tide_heatmap(comp_csv: Path, figdir: Path) -> None:
    df = pd.read_csv(comp_csv)
    scopes = ["事件日-全部", "事件日-沿海(low_coastal)", "事件日-内陆(low_coastal=0)"]
    scope_short = {"事件日-全部": "全部（n=935,220）",
                   "事件日-沿海(low_coastal)": "沿海 low_coastal（n=331,440）",
                   "事件日-内陆(low_coastal=0)": "内陆（n=603,780）"}
    rains = ["雨低", "雨中", "雨高"]
    tides = ["潮低", "潮中", "潮高"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    for j, sc in enumerate(scopes):
        sub = df[df["scope"] == sc].set_index(["rain_bin", "tide_bin"])
        for i, (valcol, fmt, title, cmap) in enumerate((
                ("pos_rate", lambda v: f"{v*100:.1f}%", "实际正例率（弱监督标签层）", "YlOrRd"),
                ("mean_p", lambda v: f"{v:.3f}", "平均模型评分（dual_xgb）", "Blues"))):
            mat = np.full((3, 3), np.nan)
            ann = np.empty((3, 3), dtype=object)
            for r, rb in enumerate(rains):
                for c, tb in enumerate(tides):
                    if (rb, tb) in sub.index:
                        v = sub.loc[(rb, tb), valcol]
                        n = int(sub.loc[(rb, tb), "n"])
                        mat[r, c] = v
                        ann[r, c] = f"{fmt(v)}\nn={n:,}"
            ax = axes[i, j]
            im = ax.imshow(mat * (100 if valcol == "pos_rate" else 1), cmap=cmap,
                           aspect="auto", vmin=0)
            ax.set_xticks(range(3), tides)
            ax.set_yticks(range(3), rains)
            for r in range(3):
                for c in range(3):
                    ax.text(c, r, ann[r, c], ha="center", va="center", fontsize=8.5)
            ax.set_title(f"{scope_short[sc]}\n{title}", fontsize=9.5)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("雨量×潮位分组（事件日，雨/潮各三分位）——标签层聚集存在但模型评分与之不同调；"
                 "弱监督标签下不构成物理交互检验", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figdir / "fig_paper_rain_tide_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 任务 2c/3：测试期 vs 全期网格风险地图（口径区分）
# ---------------------------------------------------------------------------
def grid_risk_tables(pf: pd.DataFrame, pn: pd.DataFrame, gis: pd.DataFrame) -> pd.DataFrame:
    g = pf.groupby("grid_id", observed=True).agg(
        n_rows_test=("p_dual_xgb", "size"),
        mean_p_test_dual=("p_dual_xgb", "mean"),
        max_p_test_dual=("p_dual_xgb", "max"),
    )
    pn_mean = pn.groupby("grid_id", observed=True)["p_no_tide_xgb"].mean()
    g["mean_p_test_notide"] = g.index.map(pn_mean)
    g["max_p_test_notide"] = g.index.map(
        pn.groupby("grid_id", observed=True)["p_no_tide_xgb"].max())
    g["rank_test_dual"] = g["mean_p_test_dual"].rank(ascending=False, method="first").astype(int)
    g["rank_test_notide"] = g["mean_p_test_notide"].rank(ascending=False, method="first").astype(int)
    g = g.join(gis.set_index(gis["grid_id"].astype(str))[["lon_wgs84", "lat_wgs84",
                                                          "mean_risk_all", "rank_mean_risk"]])
    g["top20_test_dual"] = g["rank_test_dual"] <= 20
    g["top20_test_notide"] = g["rank_test_notide"] <= 20
    return g.reset_index().rename(columns={"grid_id": "grid_id"})


def fig_grid_maps(gr: pd.DataFrame, figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    panels = [
        ("mean_p_test_dual", "(a) 测试期平均评分（2023–2026.09，dual_xgb，test 采样槽）"),
        ("mean_risk_all", "(b) 全期平均评分（2018–2026.09，含训练期，正式训练 GIS 产物）"),
    ]
    for ax, (col, title) in zip(axes, panels):
        sc = ax.scatter(gr["lon_wgs84"], gr["lat_wgs84"], c=gr[col], s=7,
                        cmap="YlOrRd", linewidths=0)
        top = gr[gr["top20_test_dual"]]
        ax.scatter(top["lon_wgs84"], top["lat_wgs84"], s=26, facecolors="none",
                   edgecolors="#1f77b4", linewidths=1.1, label="测试期 Top20（蓝圈）")
        ax.set(title=title, xlabel="经度", ylabel="纬度", aspect="equal")
        ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="平均风险评分")
    fig.suptitle("500 米网格风险分布：两图口径不同——(a) 仅时间外测试期，(b) 含训练期数据；"
                 "评分为弱监督标签下的模型风险评分，非已验证的真实水浸概率", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figdir / "fig_paper_grid_risk_test_vs_full.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 任务 2d：排名稳定性（dual vs no_tide）
# ---------------------------------------------------------------------------
def fig_rank_stability(gr: pd.DataFrame, spearman: float, overlap: int, figdir: Path) -> None:
    both = gr[gr["top20_test_dual"] & gr["top20_test_notide"]]
    only_d = gr[gr["top20_test_dual"] & ~gr["top20_test_notide"]]
    only_n = gr[~gr["top20_test_dual"] & gr["top20_test_notide"]]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(gr["rank_test_dual"], gr["rank_test_notide"], s=4, c="#bbbbbb", alpha=0.5,
               label=f"全部 5,517 格（Spearman {spearman:.3f}）")
    ax.scatter(only_d["rank_test_dual"], only_d["rank_test_notide"], s=30, c="#ff7f0e",
               label=f"仅 dual Top20（{len(only_d)} 格）")
    ax.scatter(only_n["rank_test_dual"], only_n["rank_test_notide"], s=30, c="#2ca02c",
               label=f"仅 no_tide Top20（{len(only_n)} 格）")
    ax.scatter(both["rank_test_dual"], both["rank_test_notide"], s=42, c="#1f77b4",
               label=f"两者均 Top20（{len(both)} 格，重合 {overlap}/20）")
    lim = (0, 120)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    ax.set(xlim=lim, ylim=lim, xlabel="dual_xgb 测试期排名", ylabel="no_tide_xgb 测试期排名",
           title="去潮位前后空间风险排名稳定性（测试期格内均值口径）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig_paper_rank_stability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 任务 2e：SHAP top20 条形图
# ---------------------------------------------------------------------------
def fig_shap(figdir: Path) -> None:
    df = pd.read_csv(SHAP_CSV).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.barh(df["feature"], df["mean_abs_shap_logodds"], color="#1f77b4", alpha=0.85)
    ax.set(xlabel="平均 |SHAP|（log-odds，精确 TreeSHAP，test）",
           title="dual_xgb 特征重要度 Top20（SHAP）\n静态地形与前期降雨主导；潮位特征 tide_pred_max_m 列第 11、非主导")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(figdir / "fig_paper_shap_top20.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pkg = ROOT / f"outputs/paper_package_b1_{ts}"
    figdir = pkg / "figures"
    tbldir = pkg / "tables"
    figdir.mkdir(parents=True)
    tbldir.mkdir(parents=True)
    print(f"[out] {pkg}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok_sha = sha256(DATA) == manifest["outputs"]["main"]["sha256"]
    if not ok_sha:
        print("[ABORT] SHA256 与 manifest 不一致")
        return 2
    print(f"[0] sha match=True（只读整理，不训练）")

    formal = json.loads((FORMAL_DIR / "metrics.json").read_text(encoding="utf-8"))
    supp = json.loads((SUPP_DIR / "metrics.json").read_text(encoding="utf-8"))
    supp_ev = json.loads((ROOT / "outputs/final_verification_spatial_b1_20261026-030632/metrics.json")
                         .read_text(encoding="utf-8"))

    bundles_dir = FORMAL_DIR / "bundles"
    # ---- 任务 1：对比表 -------------------------------------------------------
    comp_tbl = build_comparison_table(formal, supp, bundles_dir)
    comp_tbl.to_csv(tbldir / "final_model_comparison.csv", index=False)
    print(f"[1] comparison rows={len(comp_tbl)}")

    # ---- 读预测（只读）--------------------------------------------------------
    pf = pd.read_parquet(PRED_FORMAL)
    pn_frame = pd.read_parquet(PRED_NOTIDE)
    print(f"[2] preds formal={len(pf):,} no_tide={len(pn_frame):,}")

    # ---- 任务 3：测试期网格风险表 --------------------------------------------
    gis = pd.read_csv(GIS_CSV)
    gr = grid_risk_tables(pf, pn_frame, gis)
    gr.to_csv(tbldir / "grid_risk_test_period_500m.csv", index=False)
    spearman = float(gr["mean_p_test_dual"].corr(gr["mean_p_test_notide"], method="spearman"))
    overlap = int((gr["top20_test_dual"] & gr["top20_test_notide"]).sum())
    print(f"[3] grid table rows={len(gr)} spearman={spearman:.6f} top20_overlap={overlap}")

    # ---- 任务 2：图件 ----------------------------------------------------------
    stats = fig_roc_pr(pf, pn_frame, figdir)
    print(f"[2] roc/pr curves done")
    fig_rain_tide_heatmap(
        ROOT / "outputs/final_verification_spatial_b1_20261026-030632/rain_tide_composite_recheck.csv",
        figdir)
    print(f"[2] rain-tide heatmap done")
    fig_grid_maps(gr, figdir)
    fig_rank_stability(gr, spearman, overlap, figdir)
    fig_shap(figdir)
    print(f"[2] maps/rank/shap done")

    # ---- 汇集既有表格与原图 ----------------------------------------------------
    shutil.copy(SUPP_DIR / "rain_tide_composite_event_days.csv", tbldir / "rain_tide_composite_event_days_supp.csv")
    shutil.copy(ROOT / "outputs/final_verification_spatial_b1_20261026-030632/rain_tide_composite_recheck.csv",
                tbldir / "rain_tide_composite_recheck.csv")
    shutil.copy(SUPP_DIR / "threshold_sensitivity_test.csv", tbldir / "threshold_sensitivity_test.csv")
    shutil.copy(SUPP_DIR / "coastal_groups_test.csv", tbldir / "coastal_groups_test.csv")
    orig = pkg / "formal_originals"
    orig.mkdir(exist_ok=True)
    for f in (FORMAL_DIR / "figures").glob("*.png"):
        shutil.copy(f, orig / f.name)

    # ---- 落盘 run 指标 ---------------------------------------------------------
    write_json(pkg / "metrics.json", {
        "created_at": ts, "run_kind": "paper_package_b1", "read_only": True,
        "data_sha256_ok": True,
        "comparison_table": comp_tbl.to_dict("records"),
        "roc_pr_recomputed": stats,
        "rank_stability": {"spearman_mean": round(spearman, 6), "top20_overlap": overlap},
        "figure_files": sorted(f.name for f in figdir.glob("*.png")),
        "total_seconds": round(time.time() - t0, 1),
    })
    print(f"\n[done] {round(time.time()-t0,1)}s → {pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
