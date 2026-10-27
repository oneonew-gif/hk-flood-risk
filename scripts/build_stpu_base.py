#!/usr/bin/env python3
"""T3 · STPU 区域级底表构建（只读输入，独立输出）

输入（全部只读，不修改）：
  data/delivery_v10_20261025/grid_500m_to_stpu.csv        网格→主 STPU 映射 + STPU 级人口属性
  data/delivery_v10_20261025/grid_500m_stpu_overlap.csv   网格×STPU 面积权重长表
  data/delivery_v10_20261025/qc_grid_stpu_map.json        A 方映射 QC
  outputs/paper_package_b1_20261026-122643/tables/grid_risk_test_period_500m.csv  冻结测试期网格评分
输出：outputs/t3_stpu_base_20261027/

原则：不生成复合指数、不生成正式排名、不构造几何；风险/人口暴露/脆弱性分列展示。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "data/delivery_v10_20261025"
OUT = ROOT / "outputs/t3_stpu_base_20261027"
OUT.mkdir(parents=True, exist_ok=True)

MAP_CSV = DELIV / "grid_500m_to_stpu.csv"
OVL_CSV = DELIV / "grid_500m_stpu_overlap.csv"
QC_JSON = DELIV / "qc_grid_stpu_map.json"
RISK_CSV = ROOT / "outputs/paper_package_b1_20261026-122643/tables/grid_risk_test_period_500m.csv"
STATIC_CSV = DELIV / "grid_500m_static.csv"

GRID_AREA_KM2 = 0.25  # 500 m × 500 m


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    m = pd.read_csv(MAP_CSV)
    ovl = pd.read_csv(OVL_CSV)
    risk = pd.read_csv(RISK_CSV)
    static = pd.read_csv(STATIC_CSV)
    qc = json.loads(QC_JSON.read_text())

    # ---------- 1. 基础一致性 ----------
    assert set(risk.grid_id) == set(m.grid_id), "风险表与映射表网格集合不一致"
    assert not m.grid_id.duplicated().any(), "映射表存在重复 grid_id"
    assert not ovl.duplicated(["grid_id", "stpug"]).any(), "权重长表存在重复 (grid_id, stpug)"

    # STPU 级属性在网格表中必须是常量（否则按格求和会重复计数）
    attr_cols = [
        "pop_total", "pop_density_km2", "pct_aged_65plus", "household_income_median",
        "social_vulnerability_index", "vulnerability_rank",
        "exposure_population", "exposure_pop_above_median", "stpug_chi",
    ]
    attr_const = {c: int(m.groupby("stpug")[c].nunique().max()) for c in attr_cols}
    assert all(v == 1 for v in attr_const.values()), f"属性非 STPU 常量: {attr_const}"

    # ---------- 2. 权重核验 ----------
    fsum = ovl.groupby("grid_id").overlap_frac.sum()
    zero_weight_grids = sorted(fsum[fsum < 1e-9].index.tolist())
    grids_sum_lt1 = int((fsum < 1 - 1e-3).sum())
    # 主映射表 frac 为 3 位小数，长表为 4 位；差异应仅来自舍入
    pri = m[["grid_id", "stpug", "stpu_overlap_frac"]].merge(
        ovl, on=["grid_id", "stpug"], how="left", validate="one_to_one"
    )
    frac_abs_diff = (pri.stpu_overlap_frac - pri.overlap_frac).abs()
    rounded_zero_rows = int(((pri.stpu_overlap_frac < 1e-9) & (pri.overlap_frac > 0)).sum())

    # ---------- 3. 风险按面积权重聚合 ----------
    df = ovl.merge(
        risk[["grid_id", "mean_p_test_dual", "max_p_test_dual", "mean_p_test_notide", "mean_risk_all"]],
        on="grid_id", how="left", validate="many_to_one",
    )
    assert df[["mean_p_test_dual", "mean_risk_all"]].notna().all().all(), "风险值存在缺失"
    df["w"] = df.overlap_frac
    df["w_p"] = df.w * df.mean_p_test_dual
    df["w_pn"] = df.w * df.mean_p_test_notide
    df["w_pall"] = df.w * df.mean_risk_all

    agg = df.groupby("stpug").agg(
        n_grids_overlap=("grid_id", "nunique"),
        weight_sum=("w", "sum"),
        _wp=("w_p", "sum"),
        _wpn=("w_pn", "sum"),
        _wpall=("w_pall", "sum"),
    )
    agg["risk_test_dual_wmean"] = agg._wp / agg.weight_sum
    agg["risk_test_notide_wmean"] = agg._wpn / agg.weight_sum
    agg["risk_all_wmean"] = agg._wpall / agg.weight_sum
    agg["stpu_area_covered_km2_est"] = agg.weight_sum * GRID_AREA_KM2

    # 非加权对照（若严格按面积加权与简单平均差异大，说明权重不可忽略）
    umean = df.groupby("stpug").mean_p_test_dual.mean().rename("risk_test_dual_umean")
    umax = df.groupby("stpug").max_p_test_dual.max().rename("risk_test_dual_max")
    agg = agg.join([umean, umax])

    # ---------- 4. 映射状态 ----------
    primary = m.drop_duplicates("stpug").set_index("stpug")
    base = agg.join(
        primary[[
            "stpug_chi", "pop_total", "pop_density_km2", "pct_aged_65plus",
            "household_income_median", "social_vulnerability_index",
            "vulnerability_rank", "exposure_population",
        ]],
        how="left",
    )
    base["has_primary_mapping"] = base.index.isin(primary.index)
    base["has_population_attr"] = base.pop_total.notna()
    base["zero_weight_grid_count"] = base.index.map(
        df[df.w < 1e-9].groupby("stpug").grid_id.nunique()
    ).fillna(0).astype(int)
    base = base.reset_index().rename(columns={"index": "stpug"})

    base["pop_density_km2"] = base.pop_density_km2.round(1)
    base["pct_aged_65plus"] = base.pct_aged_65plus.round(2)
    for c in ["risk_test_dual_wmean", "risk_test_notide_wmean", "risk_all_wmean",
              "risk_test_dual_umean", "risk_test_dual_max", "weight_sum",
              "stpu_area_covered_km2_est"]:
        base[c] = base[c].round(6)

    # 明确声明：不生成复合指数 / 优先级 / 排名
    base = base[[
        "stpug", "stpug_chi", "has_primary_mapping", "has_population_attr",
        "n_grids_overlap", "weight_sum", "stpu_area_covered_km2_est", "zero_weight_grid_count",
        "risk_test_dual_wmean", "risk_test_dual_umean", "risk_test_dual_max",
        "risk_test_notide_wmean", "risk_all_wmean",
        "pop_total", "pop_density_km2", "pct_aged_65plus",
        "household_income_median", "social_vulnerability_index", "vulnerability_rank",
        "exposure_population",
    ]].sort_values("stpug").reset_index(drop=True)
    base.to_csv(OUT / "stpu_base_table.csv", index=False, encoding="utf-8-sig")

    # ---------- 5. 交叉校验：与旧候选清单 stpu_score_test 对比 ----------
    old_path = ROOT / "outputs/paper_tables_fill_20261026/app_stpu_candidate_checklist.csv"
    valid = {}
    if old_path.exists():
        old = pd.read_csv(old_path)[["stpug", "stpu_score_test", "n_grids", "sum_frac"]]
        cmp = base.merge(old, on="stpug", how="inner")
        diff = (cmp.risk_test_dual_wmean - cmp.stpu_score_test).abs()
        valid = {
            "n_compared": int(len(cmp)),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "n_grids_match": bool((cmp.n_grids_overlap == cmp.n_grids.astype(int)).all()),
            "weight_sum_match_max_diff": float((cmp.weight_sum - cmp.sum_frac).abs().max()),
        }

    # ---------- 6. 字段字典 / 数据来源 / QC ----------
    fdict = [
        ("stpug", "string", "—", "STPU 统计小区唯一编码（非官方社区名称）", "A 方映射表", "可用"),
        ("stpug_chi", "string", "—", "STPU 组合编码中文名（如「932, 934 及 941 - 943」），非官方社区名", "A 方映射表", "可用（仅编码名）"),
        ("has_primary_mapping", "bool", "—", "该 STPU 是否拥有至少一个网格的主映射（207/211）", "派生", "可用"),
        ("has_population_attr", "bool", "—", "是否有 2021 普查人口属性（4 个 STPU 无）", "派生", "可用"),
        ("n_grids_overlap", "int", "个", "与该 STPU 有正面积重叠的网格数（含非主映射网格）", "长表聚合", "可用"),
        ("weight_sum", "float", "网格面积单位", "Σ overlap_frac；×0.25 km² = 估算覆盖面积", "长表聚合", "可用"),
        ("stpu_area_covered_km2_est", "float", "km²", "weight_sum × 0.25；非 STPU 总面积（无边界，无法算覆盖率）", "派生", "可用（估算）"),
        ("zero_weight_grid_count", "int", "个", "权重为 0 的网格数（极小重叠或边界外）", "派生", "可用"),
        ("risk_test_dual_wmean", "float", "概率评分 [0,1]", "面积加权平均（Σf·p/Σf），dual_xgb，测试期 2023-01~2026-09", "冻结评分 + 长表", "可用（弱监督相对评分）"),
        ("risk_test_dual_umean", "float", "概率评分 [0,1]", "简单算术平均（对照，不含面积权重）", "冻结评分", "可用（对照）"),
        ("risk_test_dual_max", "float", "概率评分 [0,1]", "STPU 内网格测试期最大评分", "冻结评分", "可用"),
        ("risk_test_notide_wmean", "float", "概率评分 [0,1]", "面积加权平均，no_tide_xgb 消融模型", "补充实验评分", "待验证（消融模型，仅对比用）"),
        ("risk_all_wmean", "float", "概率评分 [0,1]", "面积加权平均，全期 train+val+test 平均（非黄金标准）", "冻结 GIS 产物", "可用（口径须标注）"),
        ("pop_total", "int", "人", "2021 普查常住人口，STPU 级常量（不是网格人口）", "A 方映射表", "可用（禁止按格求和）"),
        ("pop_density_km2", "float", "人/km²", "2021 普查人口密度（STPU 级常量）", "A 方映射表", "可用"),
        ("pct_aged_65plus", "float", "%", "65 岁及以上人口占比（2021 普查）", "A 方映射表", "可用"),
        ("household_income_median", "int", "HK$ / 月", "家庭收入中位数（2021 普查，待 A 方确认单位为月收入）", "A 方映射表", "待验证（单位）"),
        ("social_vulnerability_index", "float", "无量纲", "社会脆弱性指数；计算方法未公开，无法复算", "A 方映射表", "待验证（不参与正式分析）"),
        ("vulnerability_rank", "int", "名次 1–211", "脆弱性排名；在 211 个 STPU 上计算（207 有属性，含并列）", "A 方映射表", "待验证（随 SVI 方法）"),
        ("exposure_population", "int", "人", "暴露人口；与 pop_total 逐值恒等，非独立指标", "A 方映射表", "不可用（重复计数陷阱）"),
    ]
    pd.DataFrame(fdict, columns=["field", "dtype", "unit", "definition", "source", "status"]).to_csv(
        OUT / "field_dictionary.csv", index=False, encoding="utf-8-sig"
    )

    sources = [
        ("网格→STPU 主映射 + 人口属性", MAP_CSV, 5517, "2021 普查属性；映射为 A 方 v10 交付", sha256(MAP_CSV)),
        ("网格×STPU 面积权重长表", OVL_CSV, 7703, "A 方 v10 交付（4 位小数）", sha256(OVL_CSV)),
        ("映射 QC", QC_JSON, None, "A 方自检报告（n_stpu 211 / primary 207）", sha256(QC_JSON)),
        ("测试期网格风险评分", RISK_CSV, 5517, "冻结论文包 b1（dual_xgb 2023-01~2026-09）", sha256(RISK_CSV)),
        ("网格静态属性", STATIC_CSV, 5517, "A 方 v10 交付（WorldPop / GIS 派生）", sha256(STATIC_CSV)),
    ]
    pd.DataFrame(sources, columns=["dataset", "path", "rows", "definition", "sha256"]).to_csv(
        OUT / "data_source_table.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {p.name: sha256(p) for p in [MAP_CSV, OVL_CSV, QC_JSON, RISK_CSV, STATIC_CSV]},
        "stpu_total": int(base.stpug.nunique()),
        "stpu_primary": int(base.has_primary_mapping.sum()),
        "stpu_with_pop": int(base.has_population_attr.sum()),
        "stpu_without_primary": sorted(base.loc[~base.has_primary_mapping, "stpug"].tolist()),
        "grids": int(m.grid_id.nunique()),
        "overlap_rows": int(len(ovl)),
        "attr_constant_check": attr_const,
        "weight_check": {
            "grids_frac_sum_zero": zero_weight_grids,
            "grids_frac_sum_lt_1": grids_sum_lt1,
            "primary_frac_max_abs_diff_vs_longtable": float(frac_abs_diff.max()),
            "primary_rows_rounded_to_zero": rounded_zero_rows,
        },
        "risk_validation_vs_old_checklist": valid,
        "population_check": {
            "pop_total_sum_all_stpu": int(primary.pop_total.sum()),
            "census_total_2021": int(qc["pop_reconciliation"]["census_total"]),
            "exposure_equals_pop_total": bool((primary.exposure_population == primary.pop_total).all()),
            "worldpop_corr": qc["pop_reconciliation"]["pearson_r_by_stpu"],
        },
        "svi": {
            "min": float(primary.social_vulnerability_index.min()),
            "max": float(primary.social_vulnerability_index.max()),
            "mean": round(float(primary.social_vulnerability_index.mean()), 4),
            "rank_unique": int(primary.vulnerability_rank.nunique()),
            "recomputable": False,
        },
    }
    (OUT / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
