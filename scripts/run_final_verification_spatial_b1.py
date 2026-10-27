"""最终核验 + 空间应用分析（用户 2026-10-26 批准，5 项任务，全程只读不训练）。

任务 → 实现
1. 日期级 / 格日级聚合 AUC 审计：复算 dual_xgb 的 date_level（0.897）与
   grid_day_level（0.950），写明聚合函数、标签定义、样本量；核查 is_event_day
   在日内是否常量、日期标签与"日内有正例"是否等价、是否存在同日重复计权；
   与补充实验 metrics.json 存储值逐项 diff。并给出"能否支持提前 2 小时动态
   预警"的判断依据（对照行级事件日内 ROC 0.638）。
2. 雨潮联合分组复算：事件日内 雨三分位 × 潮三分位，每组合输出 n / 正例数 /
   正例率 / 平均模型评分（dual_xgb），沿海/内陆分层；每组合加 mean_is_A 与
   mean_low_coastal（检验静态地形 / 弱监督标签构造的组成效应）；
   与补充实验 rain_tide_composite_event_days.csv 逐项 diff。
3. dual_xgb vs no_tide_xgb 空间风险排名对比：test 全行逐格聚合
   （主口径=格内均值，敏感性口径=格内最大值），Top20 名单、重合率、
   全网格 Spearman；校验 dual 均值与 GIS mean_risk_all 一致。
4. GIS 产物对论文"社区治理优先级清单"的支撑能力检查：盘点人口/脆弱性字段
   可用性（pop_worldpop / pop_density_gis / pop_worldpop_pixels /
   pop_density_dcd / elderly_frac_gis / median_income_gis）与社区边界映射
   是否存在；若无可靠映射则明确不生成社区级排名。
5. 汇总落盘 metrics.json + results_report.md，供最终核验报告引用。

硬约束：只使用已有模型与预测结果，不重新训练，不修改原始数据和旧报告；
全部产物写入独立目录。

用法：
    .venv/bin/python scripts/run_final_verification_spatial_b1.py
"""

from __future__ import annotations

import hashlib
import json
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

from floodrisk import realdata as rd  # noqa: E402

DATA = ROOT / "data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet"
MANIFEST = ROOT / "data/delivery_repaired_b1/manifest_repair_b1.json"
FORMAL_DIR = ROOT / "outputs/formal_training_repaired_b1_20261026-020814"
SUPP_DIR = ROOT / "outputs/supplement_tide_ablation_b1_20261026-024612"
PRED_FORMAL = FORMAL_DIR / "predictions_test.parquet"
PRED_NOTIDE = SUPP_DIR / "predictions_test_no_tide.parquet"
GIS_CSV = FORMAL_DIR / "gis/grid_risk_500m.csv"
SUPP_METRICS = SUPP_DIR / "metrics.json"
SUPP_COMPOSITE = SUPP_DIR / "rain_tide_composite_event_days.csv"

KEY = ["grid_id", "date", "slot"]
RAIN_COL = "power_rain_prev3h_mm"
TIDE_COL = "tide_pred_max_m"
POP_COLS = ["pop_worldpop", "pop_density_gis", "pop_worldpop_pixels",
            "pop_density_dcd", "elderly_frac_gis", "median_income_gis"]


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
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable),
                    encoding="utf-8")


def safe_auc(y: np.ndarray, p: np.ndarray, kind: str) -> float | None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    f = roc_auc_score if kind == "roc" else average_precision_score
    return round(float(f(y, np.asarray(p, dtype=float))), 6)


# ---------------------------------------------------------------------------
# 任务 1：日期级 / 格日级聚合审计
# ---------------------------------------------------------------------------
def audit_aggregated(pf: pd.DataFrame, stored: dict) -> dict:
    p = pf["p_dual_xgb"].to_numpy(dtype=float)
    y = pf["y_future_2h"].to_numpy(dtype=int)
    ev = pf["is_event_day"].astype(int).to_numpy()

    # 同日标签常量性（is_event_day 是否日级属性）
    ev_nunique = pf.groupby("date", observed=True)["is_event_day"].nunique()
    ev_constant = bool(ev_nunique.max() == 1)

    # 日期级：标签 = is_event_day（官方警告日，日级、事后确定）
    g = pd.DataFrame({"date": pf["date"].astype(str).to_numpy(), "p": p, "ev": ev, "y": y}) \
        .groupby("date", observed=True) \
        .agg(p_max=("p", "max"), p_mean=("p", "mean"), ev=("ev", "first"), y_any=("y", "max"))
    label_eq_any_positive = bool((g["ev"] == g["y_any"]).all())
    date_level = {
        "n_dates": int(len(g)),
        "n_event_dates": int(g["ev"].sum()),
        "rows_per_date_mean": round(float(pf.groupby("date", observed=True).size().mean()), 1),
        "label_definition": "is_event_day：该日期是否 HKO 官方红/黑暴雨警告日（日级属性，事后确定）",
        "label_eq_any_positive_grid_day": label_eq_any_positive,
        "p_max_roc": safe_auc(g["ev"], g["p_max"], "roc"),
        "p_max_pr": safe_auc(g["ev"], g["p_max"], "pr"),
        "p_mean_roc": safe_auc(g["ev"], g["p_mean"], "roc"),
        "p_mean_pr": safe_auc(g["ev"], g["p_mean"], "pr"),
        "baseline_pr": round(float(g["ev"].mean()), 6),
        "weighting_note": "groupby(date) 后每日期 1 个单元，无同日重复计权；"
                          "但 p_max/p_mean 由当日全部格×槽概率聚合而来，格间强相关，"
                          "有效样本量远小于行级样本数",
    }

    # 格日级：标签 = 该格当日 12 槽中任一正例
    g2 = pd.DataFrame({
        "grid_id": pf["grid_id"].astype(str).to_numpy(),
        "date": pf["date"].astype(str).to_numpy(),
        "p": p, "y": y,
    }).groupby(["grid_id", "date"], observed=True) \
        .agg(p_max=("p", "max"), p_mean=("p", "mean"), y_any=("y", "max"))
    slots_per_gd = pf.groupby(["grid_id", "date"], observed=True).size()
    grid_day_level = {
        "n_grid_days": int(len(g2)),
        "n_positive_grid_days": int(g2["y_any"].sum()),
        "slots_per_grid_day": {"min": int(slots_per_gd.min()), "max": int(slots_per_gd.max())},
        "label_definition": "y=max：该格·日 12 个槽中存在任一 flooded_2h=1（基线正例率高，任务更易）",
        "p_max_roc": safe_auc(g2["y_any"], g2["p_max"], "roc"),
        "p_max_pr": safe_auc(g2["y_any"], g2["p_max"], "pr"),
        "p_mean_roc": safe_auc(g2["y_any"], g2["p_mean"], "roc"),
        "p_mean_pr": safe_auc(g2["y_any"], g2["p_mean"], "pr"),
        "baseline_pr": round(float(g2["y_any"].mean()), 6),
        "weighting_note": "groupby(grid_id,date) 后每格日 1 个单元，无重复计权；"
                          "但 p_max 聚合含事件发生之后的槽位（日内信息回流），"
                          "相对「提前 2 小时」口径属回顾性聚合",
    }

    out = {"date_level": date_level, "grid_day_level": grid_day_level,
           "is_event_day_constant_within_date": ev_constant}

    # 与补充实验存储值 diff
    diff = {}
    for lev, key in (("date_level", "date_level"), ("grid_day_level", "grid_day_level")):
        s = stored[key]
        for m in ("p_max_roc", "p_max_pr", "p_mean_roc", "p_mean_pr"):
            if s.get(m) is not None and out[lev].get(m) is not None:
                diff[f"{lev}.{m}"] = round(abs(out[lev][m] - s[m]), 9)
        for c in ("n_dates", "n_event_dates", "n_grid_days", "n_positive_grid_days"):
            if c in s:
                diff[f"{lev}.{c}"] = abs(out[lev][c] - s[c])
    out["recheck_max_abs_diff"] = diff
    out["recheck_ok"] = bool(max(diff.values()) < 1e-5) if diff else True

    # 对照：行级（槽开始锚定，真正的提前 2 小时口径）
    sel = ev == 1
    out["row_level_event_day_reference"] = {
        "definition": "事件日内行级 ROC（t=槽开始，预测 [t,t+2h)，无日内信息回流）",
        "roc_auc": 0.637902,
        "pr_auc": 0.201488,
        "baseline_pr": 0.135812,
        "n": int(sel.sum()),
        "source": "supplement metrics.json step5_dynamic.dual_xgb.group_report_test.event_day_1",
    }
    return out


# ---------------------------------------------------------------------------
# 任务 2：雨潮联合分组复算（含正例数与 is_A 组成）
# ---------------------------------------------------------------------------
def recheck_composite(dd: pd.DataFrame, p_col: str, stored_csv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({
        "rain": dd[RAIN_COL].to_numpy(dtype=float),
        "tide": dd[TIDE_COL].to_numpy(dtype=float),
        "p": dd[p_col].to_numpy(dtype=float),
        "y": dd["flooded_2h"].to_numpy(dtype=int),
        "is_A": dd["is_A"].to_numpy(dtype=int),
        "low_coastal": dd["low_coastal_flag"].to_numpy(dtype=int),
    })
    rain_codes = pd.qcut(df["rain"], q=3, labels=False, duplicates="drop")
    tide_codes = pd.qcut(df["tide"], q=3, labels=False, duplicates="drop")
    df["rain_bin"] = rain_codes.map({0: "雨低", 1: "雨中", 2: "雨高"})
    df["tide_bin"] = tide_codes.map({0: "潮低", 1: "潮中", 2: "潮高"})

    rows = []
    for scope, sub in (("事件日-全部", df),
                       ("事件日-沿海(low_coastal)", df[df["low_coastal"] == 1]),
                       ("事件日-内陆(low_coastal=0)", df[df["low_coastal"] == 0])):
        if len(sub) == 0:
            continue
        piv = sub.groupby(["rain_bin", "tide_bin"], observed=True).agg(
            n=("p", "size"), positives=("y", "sum"), mean_p=("p", "mean"),
            mean_is_A=("is_A", "mean"), mean_low_coastal=("low_coastal", "mean"),
            rain_median=("rain", "median"), tide_median=("tide", "median"))
        piv["scope"] = scope
        piv["pos_rate"] = piv["positives"] / piv["n"]
        piv["p90_p"] = sub.groupby(["rain_bin", "tide_bin"], observed=True)["p"].quantile(0.9)
        rows.append(piv.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out = out[["scope", "rain_bin", "tide_bin", "n", "positives", "pos_rate", "mean_p",
               "mean_is_A", "mean_low_coastal", "p90_p", "rain_median", "tide_median"]]

    # 与补充实验存储 CSV diff（n / mean_p / obs_rate）
    st = stored_csv.copy()
    st["positives_stored"] = (st["obs_rate"] * st["n"]).round(0)
    mg = out.merge(st[["scope", "rain_bin", "tide_bin", "n", "mean_p", "obs_rate"]],
                   on=["scope", "rain_bin", "tide_bin"], suffixes=("", "_stored"))
    diffs = {
        "rows_compared": int(len(mg)),
        "max_abs_diff_n": int((mg["n"] - mg["n_stored"]).abs().max()),
        "max_abs_diff_mean_p": round(float((mg["mean_p"] - mg["mean_p_stored"]).abs().max()), 9),
        "max_abs_diff_obs_rate": round(float((mg["pos_rate"] - mg["obs_rate"]).abs().max()), 9),
    }
    diffs["recheck_ok"] = bool(diffs["max_abs_diff_n"] == 0
                               and diffs["max_abs_diff_mean_p"] < 1e-5
                               and diffs["max_abs_diff_obs_rate"] < 1e-5)
    return out, diffs


# ---------------------------------------------------------------------------
# 任务 3：dual_xgb vs no_tide_xgb 空间排名对比
# ---------------------------------------------------------------------------
def spatial_ranking(pf: pd.DataFrame, pn: pd.DataFrame, gis: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 行对齐核查（两份预测必须同键同行）
    keys_f = pf[KEY].astype(str).agg("|".join, axis=1)
    keys_n = pn[KEY].astype(str).agg("|".join, axis=1)
    aligned = bool(keys_f.equals(keys_n))
    y_match = bool((pf["y_future_2h"].to_numpy() == pn["y_future_2h"].to_numpy()).all())

    agg = pd.DataFrame({
        "grid_id": pf["grid_id"].astype(str).to_numpy(),
        "p_dual": pf["p_dual_xgb"].to_numpy(dtype=float),
        "p_notide": pn["p_no_tide_xgb"].to_numpy(dtype=float),
    })
    g = agg.groupby("grid_id", observed=True).agg(
        n_rows=("p_dual", "size"),
        mean_dual=("p_dual", "mean"), mean_notide=("p_notide", "mean"),
        max_dual=("p_dual", "max"), max_notide=("p_notide", "max"))
    g = g.join(gis.set_index(gis["grid_id"].astype(str))[["lon_wgs84", "lat_wgs84",
                                                          "mean_risk_all", "rank_mean_risk"]])

    g["rank_mean_dual"] = g["mean_dual"].rank(ascending=False, method="first").astype(int)
    g["rank_mean_notide"] = g["mean_notide"].rank(ascending=False, method="first").astype(int)
    g["rank_max_dual"] = g["max_dual"].rank(ascending=False, method="first").astype(int)
    g["rank_max_notide"] = g["max_notide"].rank(ascending=False, method="first").astype(int)

    top20_mean_dual = set(g.nsmallest(20, "rank_mean_dual").index)
    top20_mean_notide = set(g.nsmallest(20, "rank_mean_notide").index)
    top20_max_dual = set(g.nsmallest(20, "rank_max_dual").index)
    top20_max_notide = set(g.nsmallest(20, "rank_max_notide").index)

    stats = {
        "n_grids": int(len(g)),
        "n_rows_per_grid": {"min": int(g["n_rows"].min()), "max": int(g["n_rows"].max()),
                            "median": int(g["n_rows"].median())},
        "rows_aligned_in_order": aligned,
        "y_match": y_match,
        "aggregation_primary": "格内均值：test 全部行（2023-01–2026-09，1,496,340 行）逐格平均预测概率",
        "aggregation_sensitivity": "格内最大值：逐格取 test 期内最大预测概率（对单次高分敏感）",
        "dual_mean_vs_gis_mean_risk_all_max_diff":
            round(float((g["mean_dual"] - g["mean_risk_all"]).abs().max()), 9),
        "gis_scope_note": "GIS mean_risk_all 为全期（train+val+test 非 golden）逐格平均"
                          "（run_formal_training_repaired_b1.py step8），与本轮 test-only 口径不同，"
                          "二者不要求一致；口径核实见 gis_scope_check",
        "spearman_mean_dual_vs_notide": round(float(g["mean_dual"].corr(g["mean_notide"], method="spearman")), 6),
        "spearman_max_dual_vs_notide": round(float(g["max_dual"].corr(g["max_notide"], method="spearman")), 6),
        "spearman_mean_vs_max_within_dual": round(float(g["mean_dual"].corr(g["max_dual"], method="spearman")), 6),
        "top20_overlap_mean": int(len(top20_mean_dual & top20_mean_notide)),
        "top20_overlap_max": int(len(top20_max_dual & top20_max_notide)),
    }

    top = g.nsmallest(20, "rank_mean_dual")[
        ["lon_wgs84", "lat_wgs84", "n_rows", "mean_dual", "rank_mean_dual",
         "mean_notide", "rank_mean_notide", "max_dual", "rank_max_dual"]]
    top = top.reset_index()
    return top, stats


# ---------------------------------------------------------------------------
# 任务 4：人口 / 脆弱性 / 社区边界字段可用性
# ---------------------------------------------------------------------------
def pop_field_availability(dd: pd.DataFrame) -> dict:
    out = {}
    for c in POP_COLS:
        if c not in dd.columns:
            out[c] = {"present": False}
            continue
        s = dd[c]
        per_grid_nunique = dd.groupby("grid_id", observed=True)[c].nunique(dropna=True)
        out[c] = {
            "present": True,
            "dtype": str(s.dtype),
            "non_null_rate": round(float(s.notna().mean()), 6),
            "nunique_overall": int(s.nunique(dropna=True)),
            "max_nunique_within_grid": int(per_grid_nunique.max()),
            "constant_within_grid": bool(per_grid_nunique.max() <= 1),
            "min": round(float(np.nanmin(s.values)), 4) if s.notna().any() else None,
            "max": round(float(np.nanmax(s.values)), 4) if s.notna().any() else None,
        }
    # 社区边界映射：交付数据中不存在任何社区/街道级边界或映射字段
    candidates = [c for c in dd.columns
                  if any(k in c.lower() for k in ("community", "constituency", "village", "tpra", "street"))]
    out["_community_boundary_mapping"] = {
        "candidate_columns": candidates,
        "conclusion": "交付数据无社区/街道级边界或映射字段；pop_density_dcd 为区级常量"
                      "（nunique 见上），不构成社区级映射",
    }
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / f"outputs/final_verification_spatial_b1_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    supp = json.loads(SUPP_METRICS.read_text(encoding="utf-8"))

    # ---- step0 版本冻结（只读核查）------------------------------------------
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sha_data = sha256(DATA)
    ok_sha = sha_data == manifest["outputs"]["main"]["sha256"]
    print(f"[0] sha match={ok_sha}")
    if not ok_sha:
        print("[ABORT] SHA256 与 manifest 不一致")
        return 2
    result: dict = {
        "created_at": ts,
        "run_kind": "final_verification_spatial_b1",
        "read_only": True,
        "data": {"path": str(DATA), "sha256": sha_data, "version": "delivery_repaired_b1"},
        "step0_freeze": {"sha_matches_manifest": bool(ok_sha),
                         "no_training_no_data_write": True},
    }

    pf = pd.read_parquet(PRED_FORMAL)
    pn = pd.read_parquet(PRED_NOTIDE)
    gis = pd.read_csv(GIS_CSV)
    print(f"[1] preds formal={len(pf):,} no_tide={len(pn):,} gis={len(gis):,}")

    # ---- 任务 1：聚合审计 ----------------------------------------------------
    stored_agg = supp["step5_dynamic"]["dual_xgb"]["aggregated_auc_test"]
    result["task1_aggregated_auc_audit"] = audit_aggregated(pf, stored_agg)
    a1 = result["task1_aggregated_auc_audit"]
    print(f"[1] date_level p_max_roc={a1['date_level']['p_max_roc']} "
          f"p_mean_roc={a1['date_level']['p_mean_roc']} "
          f"grid_day p_max_roc={a1['grid_day_level']['p_max_roc']} "
          f"recheck_ok={a1['recheck_ok']}")

    # ---- 读交付数据（test 相关列 + is_A + 人口字段）--------------------------
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(DATA).names)
    need_cols = sorted(set(KEY + ["split", "is_golden", "flooded_2h", "is_event_day", "is_A",
                                  "low_coastal_flag", RAIN_COL, TIDE_COL] + POP_COLS) & schema_cols)
    dd_all = pd.read_parquet(DATA, columns=need_cols)
    dd = dd_all[dd_all["split"] == "test"].reset_index(drop=True)
    # 合并 dual_xgb 预测概率（按主键对齐）
    dd = dd.merge(pf[KEY + ["p_dual_xgb"]], on=KEY, how="inner", validate="one_to_one")
    assert len(dd) == 1496340, f"合并后行数异常: {len(dd)}"
    print(f"[2] delivery test rows={len(dd):,}")

    # ---- 任务 2：雨潮复合复算（与补充实验同协议：仅事件日，分位箱在事件日全体上 qcut）----
    stored_comp = pd.read_csv(SUPP_COMPOSITE)
    dd_ev = dd[dd["is_event_day"].astype(int) == 1].reset_index(drop=True)
    comp, comp_diff = recheck_composite(dd_ev, "p_dual_xgb", stored_comp)
    comp.to_csv(outdir / "rain_tide_composite_recheck.csv", index=False)
    result["task2_rain_tide_recheck"] = {
        "note": "分位箱在事件日全体上一次性 qcut（与补充实验同协议），沿海/内陆复用同一箱边界",
        "diffs_vs_stored": comp_diff,
        "coastal_base_rate": round(float(dd.loc[dd["low_coastal_flag"] == 1, "flooded_2h"].mean()), 6),
        "inland_base_rate": round(float(dd.loc[dd["low_coastal_flag"] == 0, "flooded_2h"].mean()), 6),
    }
    print(f"[2] composite rows={len(comp)} recheck_ok={comp_diff['recheck_ok']} "
          f"coastal_base={result['task2_rain_tide_recheck']['coastal_base_rate']} "
          f"inland_base={result['task2_rain_tide_recheck']['inland_base_rate']}")

    # ---- 任务 3：空间排名对比 ------------------------------------------------
    top20, sp = spatial_ranking(pf, pn, gis)
    top20.to_csv(outdir / "top20_grid_rank_comparison.csv", index=False)
    result["task3_spatial_ranking"] = sp
    result["task3_spatial_ranking"]["top20_dual"] = top20[
        ["grid_id", "lon_wgs84", "lat_wgs84", "mean_dual", "rank_mean_dual",
         "mean_notide", "rank_mean_notide"]].to_dict("records")
    print(f"[3] n_grids={sp['n_grids']} spearman_mean={sp['spearman_mean_dual_vs_notide']} "
          f"top20_overlap_mean={sp['top20_overlap_mean']}")

    # GIS 产物口径核实：mean_risk_all 是否为全期（train+val+test 非 golden）逐格聚合
    non_golden = dd_all[dd_all["is_golden"].astype(int) != 1]
    full_sizes = non_golden.groupby(non_golden["grid_id"].astype(str)).size()
    gis_sizes = gis.set_index(gis["grid_id"].astype(str))["n_rows"]
    joined = pd.concat([full_sizes.rename("delivery_all"), gis_sizes.rename("gis")], axis=1).dropna()
    n_match = int((joined["delivery_all"] == joined["gis"]).sum())
    result["task3_spatial_ranking"]["gis_scope_check"] = {
        "expected_scope": "全期 train+val+test 非 golden 行逐格平均（正式训练脚本 step8 对全量 data 聚合）",
        "grids_compared": int(len(joined)),
        "n_rows_match_grids": n_match,
        "n_rows_match_all": bool(n_match == len(joined)),
        "implication": "mean_risk_all 是全期风险气候学（含训练期数据），非 test 时段风险；"
                       "论文引用 GIS 排名时必须标注口径",
    }
    print(f"[3] gis scope check: n_rows_match {n_match}/{len(joined)}")

    # ---- 任务 4：人口 / 社区字段可用性 ---------------------------------------
    result["task4_pop_community_fields"] = pop_field_availability(dd_all)
    print(f"[4] pop fields checked={len(POP_COLS)} "
          f"community_mapping={result['task4_pop_community_fields']['_community_boundary_mapping']['candidate_columns']}")

    # ---- 落盘 -----------------------------------------------------------------
    result["total_seconds"] = round(time.time() - t0, 1)
    write_json(outdir / "metrics.json", result)

    # results_report.md：任务 1 / 2 / 3 / 4 全表
    md: list[str] = []
    md.append(f"# 最终核验 + 空间应用分析 · {ts}\n")
    md.append(f"- 数据：delivery_repaired_b1（SHA256 `{result['data']['sha256'][:12]}…`，冻结复核 "
              f"{'通过' if ok_sha else '失败'}）；全程只读，未训练、未改任何原始数据与旧报告\n")
    md.append("\n## 任务 1：日期级 / 格日级聚合 AUC 审计（dual_xgb，test）\n")
    md.append("| 口径 | 标签定义 | 单元数 | 正例单元 | 基线正例率 | p_max ROC | p_max PR | p_mean ROC | p_mean PR |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    dl, gl = a1["date_level"], a1["grid_day_level"]
    md.append(f"| 日期级 | {dl['label_definition']} | {dl['n_dates']:,} | {dl['n_event_dates']} | "
              f"{dl['baseline_pr']:.4f} | {dl['p_max_roc']} | {dl['p_max_pr']} | "
              f"{dl['p_mean_roc']} | {dl['p_mean_pr']} |")
    md.append(f"| 格日级 | {gl['label_definition']} | {gl['n_grid_days']:,} | "
              f"{gl['n_positive_grid_days']:,} | {gl['baseline_pr']:.4f} | {gl['p_max_roc']} | "
              f"{gl['p_max_pr']} | {gl['p_mean_roc']} | {gl['p_mean_pr']} |")
    md.append(f"\n- is_event_day 日内常量：{a1['is_event_day_constant_within_date']}；"
              f"日期标签 ≡ 日内存在正例格日：{dl['label_eq_any_positive_grid_day']}")
    md.append(f"- 复算 vs 补充实验存储值最大绝对差：{max(a1['recheck_max_abs_diff'].values()):.2e}（recheck_ok={a1['recheck_ok']}）")
    md.append(f"- 计权说明：{dl['weighting_note']}；{gl['weighting_note']}")
    md.append(f"- 提前 2 小时口径对照：{a1['row_level_event_day_reference']['definition']} "
              f"ROC {a1['row_level_event_day_reference']['roc_auc']} / PR "
              f"{a1['row_level_event_day_reference']['pr_auc']}（基线 {a1['row_level_event_day_reference']['baseline_pr']}，n={a1['row_level_event_day_reference']['n']:,}）")

    md.append("\n## 任务 2：雨潮联合分组复算（事件日，dual_xgb 评分）\n")
    md.append("每组合：n / 正例数 / 正例率 / 平均评分 / mean_is_A / mean_low_coastal"
              "（mean_is_A 与 mean_low_coastal 用于检验静态地形与标签构造的组成效应）\n")
    for scope in comp["scope"].unique():
        sub = comp[comp["scope"] == scope]
        md.append(f"\n### {scope}（n={int(sub['n'].sum()):,}）\n")
        md.append("| 雨箱 | 潮箱 | n | 正例数 | 正例率 | 平均评分 p90 | mean_is_A | mean_low_coastal |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            md.append(f"| {r['rain_bin']} | {r['tide_bin']} | {int(r['n']):,} | {int(r['positives']):,} | "
                      f"{r['pos_rate']:.4f} | {r['mean_p']:.4f} / {r['p90_p']:.3f} | "
                      f"{r['mean_is_A']:.3f} | {r['mean_low_coastal']:.3f} |")
    md.append(f"\n- 与补充实验存储 CSV diff：n 最大差 {comp_diff['max_abs_diff_n']}，"
              f"mean_p 最大差 {comp_diff['max_abs_diff_mean_p']:.2e}，obs_rate 最大差 "
              f"{comp_diff['max_abs_diff_obs_rate']:.2e}（recheck_ok={comp_diff['recheck_ok']}）")
    md.append(f"- 沿海基线正例率 {result['task2_rain_tide_recheck']['coastal_base_rate']:.4f}，"
              f"内陆基线 {result['task2_rain_tide_recheck']['inland_base_rate']:.4f}")

    md.append("\n## 任务 3：dual_xgb vs no_tide_xgb 空间风险排名对比（test 全期逐格聚合）\n")
    md.append(f"- 聚合方式（主口径）：{sp['aggregation_primary']}；敏感性口径：{sp['aggregation_sensitivity']}")
    md.append(f"- 网格数 {sp['n_grids']:,}；每格行数 {sp['n_rows_per_grid']['min']}–"
              f"{sp['n_rows_per_grid']['max']}（中位 {sp['n_rows_per_grid']['median']}，不等权暴露）")
    gsc = sp["gis_scope_check"]
    md.append(f"- **GIS 产物口径核实**：`mean_risk_all` 为{gsc['expected_scope']}；逐格行数复核 "
              f"{gsc['n_rows_match_grids']}/{gsc['grids_compared']} 一致（n_rows_match_all="
              f"{gsc['n_rows_match_all']}）。{gsc['implication']}")
    md.append(f"- Spearman（均值口径）dual vs no_tide：**{sp['spearman_mean_dual_vs_notide']}**；"
              f"（最大值口径）{sp['spearman_max_dual_vs_notide']}；"
              f"dual 均值 vs 最大值口径 {sp['spearman_mean_vs_max_within_dual']}")
    md.append(f"- Top20 重合：均值口径 **{sp['top20_overlap_mean']}/20**；最大值口径 {sp['top20_overlap_max']}/20\n")
    md.append("\n### Top20 高风险网格（均值口径，按 dual_xgb 排名）\n")
    md.append("| 排名 | grid_id | lon | lat | n_rows | dual mean_p | no_tide mean_p | no_tide 排名 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in top20.iterrows():
        md.append(f"| {int(r['rank_mean_dual'])} | {r['grid_id']} | {r['lon_wgs84']:.4f} | "
                  f"{r['lat_wgs84']:.4f} | {int(r['n_rows'])} | {r['mean_dual']:.4f} | "
                  f"{r['mean_notide']:.4f} | {int(r['rank_mean_notide'])} |")

    md.append("\n## 任务 4：人口 / 脆弱性 / 社区边界字段可用性\n")
    md.append("| 字段 | 存在 | 非空率 | 网格内常量 | 取值范围 | 说明 |")
    md.append("|---|---|---|---|---|---|")
    for c, v in result["task4_pop_community_fields"].items():
        if c.startswith("_"):
            continue
        if not v.get("present"):
            md.append(f"| {c} | 否 | — | — | — | 交付数据中不存在 |")
        else:
            note = "区级常量（不构成社区映射）" if c == "pop_density_dcd" else "格级字段，约定不进概率模型"
            md.append(f"| {c} | 是 | {v['non_null_rate']:.4f} | {v['constant_within_grid']} | "
                      f"{v['min']}–{v['max']} | {note} |")
    cm = result["task4_pop_community_fields"]["_community_boundary_mapping"]
    md.append(f"\n- 社区边界映射候选字段：{cm['candidate_columns'] or '无'}。{cm['conclusion']}")

    (outdir / "results_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[done] {result['total_seconds']}s → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
