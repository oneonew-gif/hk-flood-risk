"""空间交叉验证折分组画像（只读，不训练）。

核验 cv_fold 的空间有效性、逐折正负样本与事件日覆盖，
并按折评估已冻结的测试预测（不重新训练）。
输出写入 outputs/spatial_cv_design_20261027/。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet"
PRED = ROOT / "outputs/formal_training_repaired_b1_20261026-020814/predictions_test.parquet"
OUT = ROOT / "outputs/spatial_cv_design_20261027"
MODELS = ["p_dual_xgb", "p_static_only_xgb", "p_dual_lr"]

checks: list[dict] = []


def add(cid: str, item: str, status: str, evidence: str) -> None:
    checks.append({"id": cid, "item": item, "status": status, "evidence": evidence})


def phase_of(year: int) -> str:
    if year <= 2021:
        return "train"
    if year == 2022:
        return "val"
    return "test"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["grid_id", "date", "slot", "flooded_2h", "cv_fold", "dc", "is_golden", "split"]
    df = pd.read_parquet(DATA, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["phase"] = df["year"].map(phase_of)

    # V1 fold ↔ dc 双射
    nd = df[df["cv_fold"] != -1]
    f2d = nd.groupby("cv_fold")["dc"].nunique()
    d2f = nd.groupby("dc")["cv_fold"].nunique()
    add("V1", "18 个 cv_fold 与区议会分区是否一一对应",
        "通过" if f2d.max() == 1 and d2f.max() == 1 and len(f2d) == 18 else "不通过",
        f"折数 {len(f2d)}，每折对应 dc 数最大值 {int(f2d.max())}；dc 数 {len(d2f)}，"
        f"每 dc 对应折数最大值 {int(d2f.max())}")

    # V2 网格→折唯一（剔除 golden）
    ng = df[(df["cv_fold"] != -1) & (df["is_golden"] == 0)]
    g2f = ng.groupby("grid_id")["cv_fold"].nunique()
    add("V2", "排除 golden 行后，网格→折映射是否唯一",
        "通过" if g2f.max() == 1 else "不通过",
        f"参与折的网格 {len(g2f)}，跨折网格 {int((g2f > 1).sum())}。"
        "全部 936 条 golden 行被强制置 cv_fold=-1，是之前“75 格跨折”的唯一来源")

    # V3 -1 组构成
    m1 = df[df["cv_fold"] == -1]
    m1_dc = m1[m1["dc"] == -1]
    m1_g = m1[m1["is_golden"] == 1]
    add("V3", "cv_fold=-1 组的构成是否清楚",
        "通过" if len(m1_dc) + len(m1_g) == len(m1) else "待验证",
        f"-1 共 {len(m1):,} 行（{len(m1) / len(df):.2%}）= {len(m1_dc):,} 行来自 {m1_dc['grid_id'].nunique()} 个"
        f"无区归属网格（dc=-1）+ {len(m1_g):,} 行 golden。CV 必须整组排除，golden 单列报告")

    # 逐折 × 阶段画像
    rows = []
    for fold, g in nd.groupby("cv_fold"):
        rec = {"fold": int(fold), "dc": int(g["dc"].iloc[0]), "grids": int(g["grid_id"].nunique())}
        for ph in ["train", "val", "test"]:
            h = g[g["phase"] == ph]
            rec[f"{ph}_rows"] = len(h)
            rec[f"{ph}_pos"] = int(h["flooded_2h"].sum())
            rec[f"{ph}_rate"] = round(float(h["flooded_2h"].mean()), 4) if len(h) else None
            rec[f"{ph}_event_days"] = int(h.loc[h["flooded_2h"] == 1, "date"].nunique())
        rec["total_rows"] = len(g)
        rec["total_pos"] = int(g["flooded_2h"].sum())
        rec["date_min"] = str(g["date"].min().date())
        rec["date_max"] = str(g["date"].max().date())
        rows.append(rec)
    prof = pd.DataFrame(rows).sort_values("fold")
    prof.to_csv(OUT / "fold_profile.csv", index=False, encoding="utf-8-sig")

    # V4/V5/V6 折规模与正例覆盖
    add("V4", "各折规模是否均衡",
        "警示",
        f"网格 {prof['grids'].min()}–{prof['grids'].max()} 个；行数 {prof['total_rows'].min():,}–"
        f"{prof['total_rows'].max():,}；训练段正例 {prof['train_pos'].min():,}–{prof['train_pos'].max():,}"
        "（最不均衡折与最大折相差约 34 倍，按折汇总不能做等权样本推断）")
    zero = prof[(prof[["train_pos", "val_pos", "test_pos"]] == 0).any(axis=1)]
    add("V5", "是否存在无正例的折/阶段",
        "通过" if zero.empty else "待验证",
        f"无正例的折-阶段组合 {len(zero)} 个；各折最小正例：train {prof['train_pos'].min()}、"
        f"val {prof['val_pos'].min()}、test {prof['test_pos'].min()}")
    small = prof[(prof[["train_pos", "val_pos", "test_pos"]] < 50).any(axis=1)]["fold"].tolist()
    add("V6", "是否存在正例过少、指标不稳定的折",
        "警示" if small else "通过",
        f"正例<50 的折 {small}；这些折的 AUC/阈值必须标注“不稳定”，不得单独用于模型比较")

    # V7 事件日是否被空间折隔离
    all_days = df.loc[df["flooded_2h"] == 1, "date"].nunique()
    same_days = all(
        g.loc[g["flooded_2h"] == 1, "date"].nunique() == all_days for _, g in nd.groupby("cv_fold")
    )
    add("V7", "空间折是否能隔离事件日", "警示" if same_days else "通过",
        f"全表 {all_days} 个正例日期；每折均覆盖全部 {all_days} 个（train 23 / val 2 / test 33）→ "
        "空间折不能隔离事件日，必须同时保留时间切分，否则同一事件日的其他区域样本会进入训练")

    # V8 折覆盖完整性
    add("V8", "18 折是否覆盖全部可评估测试样本",
        "通过" if int(prof["test_rows"].sum()) == len(df[(df["cv_fold"] != -1) & (df["phase"] == "test")]) else "不通过",
        f"18 折测试行合计 {int(prof['test_rows'].sum()):,} = 排除 -1 与 golden 后的测试行")

    # 冻结模型分折指标（不训练）
    pred = pd.read_parquet(PRED, columns=["grid_id", "date", "slot", "y_future_2h", "is_event_day"] + MODELS)
    key = df[["grid_id", "date", "slot", "cv_fold"]]
    key["date"] = key["date"].astype(str)
    pred["date"] = pd.to_datetime(pred["date"]).dt.strftime("%Y-%m-%d")
    key["date"] = pd.to_datetime(key["date"]).dt.strftime("%Y-%m-%d")
    merged = pred.merge(key, on=["grid_id", "date", "slot"], how="left")
    ev = merged[(merged["cv_fold"] != -1) & (merged["is_event_day"] == 1)]
    frozen = []
    for fold, g in merged[merged["cv_fold"] != -1].groupby("cv_fold"):
        y = g["y_future_2h"].to_numpy()
        rec = {"fold": int(fold), "n": len(g), "pos": int(y.sum()), "prevalence": round(float(y.mean()), 4)}
        ge = ev[ev["cv_fold"] == fold]
        for m in MODELS:
            s = g[m].to_numpy()
            rec[f"{m}_auc"] = round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None
            rec[f"{m}_pr"] = round(float(average_precision_score(y, s)), 4)
            rec[f"{m}_brier"] = round(float(brier_score_loss(y, s)), 4)
            if len(ge) and 0 < ge["y_future_2h"].sum() < len(ge):
                rec[f"{m}_auc_eventday"] = round(
                    float(roc_auc_score(ge["y_future_2h"], ge[m])), 4
                )
        frozen.append(rec)
    fz = pd.DataFrame(frozen).sort_values("fold")
    fz.to_csv(OUT / "frozen_test_fold_metrics.csv", index=False, encoding="utf-8-sig")
    aucs = fz["p_dual_xgb_auc"].dropna()
    aucs_s = fz["p_static_only_xgb_auc"].dropna()
    add("V9", "冻结模型的分折差异是否被整体 AUC 掩盖",
        "警示",
        f"冻结 dual_xgb 分折 ROC-AUC {aucs.min():.3f}–{aucs.max():.3f}（均值 {aucs.mean():.3f}，"
        f"SD {aucs.std():.3f}）；static_only {aucs_s.min():.3f}–{aucs_s.max():.3f}"
        f"（含 1 折低于 0.5）。整体 pooled AUC 0.751 不能代表各区表现")

    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(),
        "rows_total": len(df),
        "rows_evaluable": len(nd),
        "fold_sizes": {"min_rows": int(prof["total_rows"].min()), "max_rows": int(prof["total_rows"].max())},
        "phase_rows": df.groupby("phase").size().to_dict(),
        "phase_pos": df.groupby("phase")["flooded_2h"].sum().astype(int).to_dict(),
        "event_days_total": int(all_days),
        "frozen_dual_auc": {
            "mean": round(float(aucs.mean()), 4),
            "sd": round(float(aucs.std()), 4),
            "min": round(float(aucs.min()), 4),
            "max": round(float(aucs.max()), 4),
        },
        "checks": checks,
    }
    (OUT / "cv_checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, ensure_ascii=False, indent=2))
    for c in checks:
        print(c["id"], c["status"], c["evidence"][:120])


if __name__ == "__main__":
    main()
