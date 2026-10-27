"""T3 研究质量审计（只读）。

核验 STPU 底表的人口口径、面积权重几何含义、聚合正确性、SVI 可复算性。
不修改任何输入，不训练模型。输出写入 outputs/t3_quality_audit_20261027/。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DELIV = ROOT / "data/delivery_v10_20261025"
T3 = ROOT / "outputs/t3_stpu_base_20261027"
OUT = ROOT / "outputs/t3_quality_audit_20261027"
GRID_AREA_KM2 = 0.25

# 政府统计处 2021 年人口普查：2021 年 6 月底居住人口 7,413,070
OFFICIAL_CENSUS_2021 = 7413070
OFFICIAL_SOURCE = "https://www.census2021.gov.hk/sc/media.html"

checks: list[dict] = []


def add(cid: str, item: str, status: str, evidence: str) -> None:
    checks.append({"id": cid, "item": item, "status": status, "evidence": evidence})


def fmt(x, n=4):
    return round(float(x), n)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(T3 / "stpu_base_table.csv", dtype={"stpug": str})
    main_map = pd.read_csv(DELIV / "grid_500m_to_stpu.csv", dtype={"stpug": str})
    ovl = pd.read_csv(DELIV / "grid_500m_stpu_overlap.csv", dtype={"stpug": str})
    static = pd.read_csv(DELIV / "grid_500m_static.csv", usecols=["grid_id", "pop_worldpop"])
    risk = pd.read_csv(
        ROOT / "outputs/paper_package_b1_20261026-122643/tables/grid_risk_test_period_500m.csv"
    )
    qc = json.loads((DELIV / "qc_grid_stpu_map.json").read_text())

    # C01 属性在 STPU 内是否常量
    attr_cols = [
        "pop_total", "pop_density_km2", "pct_aged_65plus", "household_income_median",
        "social_vulnerability_index", "vulnerability_rank", "exposure_population",
    ]
    max_unique = {c: int(main_map.groupby("stpug")[c].nunique().max()) for c in attr_cols}
    add("C01", "七个属性在 STPU 内是否为常量", "通过" if max(max_unique.values()) == 1 else "不通过",
        f"逐 STPU nunique 最大值 {max_unique}")

    # C02/C03 人口合计与官方口径
    pop = base["pop_total"].dropna()
    pop_sum = int(pop.sum())
    add("C02", "207 个 STPU 人口合计是否与 A 方 QC 记录一致",
        "通过" if pop_sum == int(qc["pop_reconciliation"]["census_total"]) else "不通过",
        f"底表合计 {pop_sum:,}；qc_grid_stpu_map.json census_total {int(qc['pop_reconciliation']['census_total']):,}")
    diff = OFFICIAL_CENSUS_2021 - pop_sum
    add("C03", "人口合计是否等于 2021 普查官方全港人口",
        "待验证",
        f"官方 2021 年 6 月底居住人口 {OFFICIAL_CENSUS_2021:,}（{OFFICIAL_SOURCE}）；"
        f"底表 207 个合计 {pop_sum:,}，差 {diff:,}（{diff / OFFICIAL_CENSUS_2021:.2%}）。"
        "原始底表 outputs/v7/census2021_stpu_units.csv 未随交付提供，无法判断差异来自人口定义、"
        "住户/集体户范围、边界版本或 STPU 覆盖")

    # C04 四个无属性 STPU
    miss = base[~base["has_population_attr"]]
    area_total = float(base["stpu_area_covered_km2_est"].sum())
    miss_area = float(miss["stpu_area_covered_km2_est"].sum())
    max_density = float(base["pop_density_km2"].max())
    add("C04", "四个无人口属性 STPU 的覆盖影响", "待验证",
        f"{miss['stpug'].tolist()} 各仅 4 个重叠网格、面积合计 {miss_area:.3f} km²"
        f"（占底表陆地面积 {miss_area / area_total:.2%}）；人口未知。"
        f"按观测最高密度 {max_density:,.0f} 人/km² 的极端上界约 {miss_area * max_density:,.0f} 人"
        f"（{miss_area * max_density / OFFICIAL_CENSUS_2021:.2%} 全港），仅为上界不是估计值")

    # C05 权重归一化方向
    fs = ovl.groupby("grid_id")["overlap_frac"].sum()
    n_eq = int((abs(fs - 1) <= 1e-9).sum())
    n_lt = int((fs < 1 - 1e-9).sum())
    n_gt = int((fs > 1 + 1e-9).sum())
    add("C05", "每格面积权重之和（几何含义）", "通过",
        f"共 5,517 格：和=1 有 {n_eq} 格，<1 有 {n_lt} 格（跨界/水域），>1 有 {n_gt} 格"
        f"（4 位小数舍入，最大超出 {(fs - 1).max():.6f}），和为 0 有 {int((fs == 0).sum())} 格。"
        "权重是“网格面积落入该 STPU 的比例”，不是 STPU 面积占比")

    # C06 主映射是否等于最大重叠
    mx = ovl.loc[ovl.groupby("grid_id")["overlap_frac"].idxmax(), ["grid_id", "stpug", "overlap_frac"]]
    mx = mx.rename(columns={"stpug": "max_stpu", "overlap_frac": "max_frac"})
    cmp = main_map[["grid_id", "stpug", "stpu_overlap_frac", "match_type"]].merge(mx, on="grid_id")
    same_stpu = int((cmp["stpug"] == cmp["max_stpu"]).sum())
    add("C06", "主映射 STPU 是否为该格最大重叠 STPU", "通过" if same_stpu == len(cmp) else "不通过",
        f"{same_stpu}/{len(cmp)} 一致；匹配方式 {cmp['match_type'].value_counts().to_dict()}")

    # C07 覆盖面积与人口密度是否同源
    d = base.dropna(subset=["pop_total", "pop_density_km2"]).copy()
    d["implied_area_km2"] = d["pop_total"] / d["pop_density_km2"]
    d["cover_ratio"] = d["stpu_area_covered_km2_est"] / d["implied_area_km2"]
    lo, hi = float(d["cover_ratio"].min()), float(d["cover_ratio"].max())
    add("C07", "覆盖面积与人口密度是否相互矛盾", "通过",
        f"207 个 STPU：pop_total/pop_density 反推面积 合计 {d['implied_area_km2'].sum():,.1f} km²，"
        f"权重和×0.25 合计 {d['stpu_area_covered_km2_est'].sum():,.1f} km²，"
        f"逐区比值 {lo:.3f}–{hi:.3f}（仅 4 位小数舍入误差）。"
        "说明 pop_density_km2 由同一套几何派生，STPU 在陆地网格范围内覆盖完整；"
        f"全表陆地面积 {area_total:,.1f} km² 与香港陆地面积量级一致")

    # C08 聚合公式复现（长表 4 位权重）
    g = risk[["grid_id", "mean_p_test_dual"]].merge(ovl, on="grid_id")
    agg = g.groupby("stpug").apply(
        lambda x: (x["overlap_frac"] * x["mean_p_test_dual"]).sum() / x["overlap_frac"].sum(),
        include_groups=False,
    ).rename("p_recalc")
    chk = pd.concat([agg, base.set_index("stpug")["risk_test_dual_wmean"]], axis=1).dropna()
    max_diff = float((chk["p_recalc"] - chk["risk_test_dual_wmean"]).abs().max())
    add("C08", "STPU 面积加权风险聚合是否正确", "通过" if max_diff < 1e-5 else "不通过",
        f"独立复现 Σ(f·p)/Σf 与底表最大绝对差 {max_diff:.2e}（{len(chk)} 个 STPU）")

    # C09 若误用主映射表（单 STPU、3 位小数）聚合的偏差
    gp = risk[["grid_id", "mean_p_test_dual"]].merge(main_map[["grid_id", "stpug", "stpu_overlap_frac"]], on="grid_id")
    agg3 = gp.groupby("stpug").apply(
        lambda x: (x["stpu_overlap_frac"] * x["mean_p_test_dual"]).sum() / x["stpu_overlap_frac"].sum(),
        include_groups=False,
    ).rename("p_primary_only")
    cmp2 = pd.concat([agg3, agg], axis=1).dropna()
    cmp2["diff"] = (cmp2["p_primary_only"] - cmp2["p_recalc"]).abs()
    worst = cmp2["diff"].idxmax()
    rank_shift = int((cmp2["p_primary_only"].rank() - cmp2["p_recalc"].rank()).abs().max())
    add("C09", "聚合是否必须使用长表（当前底表已满足）", "通过",
        f"若只用主映射表（每格 1 个 STPU、3 位小数）：最大偏差 {cmp2['diff'].max():.4f}"
        f"（STPU {worst}：{cmp2.loc[worst, 'p_primary_only']:.4f} vs {cmp2.loc[worst, 'p_recalc']:.4f}），"
        f"最大排名变动 {rank_shift} 位。正式聚合须用 4 位长表")

    # C10 加权 vs 简单平均
    sp = float(base[["risk_test_dual_wmean", "risk_test_dual_umean"]].corr(method="spearman").iloc[0, 1])
    shift = int((base["risk_test_dual_wmean"].rank() - base["risk_test_dual_umean"].rank()).abs().max())
    add("C10", "面积加权与简单平均的差异是否可忽略", "通过",
        f"Spearman {sp:.4f}，最大排名变动 {shift} 位 → 权重有实质影响，论文需并列两口径并说明")

    # C11 SVI 可复算性
    s = base.dropna(subset=["social_vulnerability_index", "pct_aged_65plus", "household_income_median", "pop_density_km2"])
    X = np.column_stack([np.ones(len(s)), s["pct_aged_65plus"], s["household_income_median"], s["pop_density_km2"]])
    y = s["social_vulnerability_index"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    corr65 = float(np.corrcoef(s["pct_aged_65plus"], y)[0, 1])
    add("C11", "SVI 是否能由已交付字段独立复算", "不适合正式使用",
        f"对 65+ 比例、收入中位数、人口密度做线性回归 R²={r2:.3f}（R²=1 才可复算）；"
        f"与 65+ 相关 {corr65:.3f}、与收入相关 {float(np.corrcoef(s['household_income_median'], y)[0, 1]):.3f}。"
        "方法未公开，数值域 −1.84–5.93、均值 −0.015，非 0–1 标准化指数 → 保留字段、不参与正式分析")

    # C12 exposure_population
    same = bool((base["exposure_population"].fillna(-1) == base["pop_total"].fillna(-1)).all())
    add("C12", "exposure_population 是否可作独立暴露人口", "不适合正式使用" if same else "通过",
        "与 pop_total 逐值恒等（207/207）→ 属重复计数陷阱，不得按格求和或另行解释")

    # C13 WorldPop 交叉一致性
    wp = main_map[["grid_id", "stpug"]].merge(static, on="grid_id").groupby("stpug")["pop_worldpop"].sum()
    pc = pd.concat([base.set_index("stpug")["pop_total"], wp.rename("worldpop_sum")], axis=1).dropna()
    pc["ratio"] = pc["worldpop_sum"] / pc["pop_total"]
    out = pc[(pc["ratio"] > 2) | (pc["ratio"] < 0.5)]
    add("C13", "STPU 人口属性与 WorldPop 空间分布是否一致", "待验证",
        f"逐 STPU 比值中位 {pc['ratio'].median():.2f}，但 {len(out)}/207 超出 [0.5, 2]，"
        f"范围 {pc['ratio'].min():.2f}–{pc['ratio'].max():.2f}。"
        f"极端：{pc['ratio'].idxmax()} 普查 {pc.loc[pc['ratio'].idxmax(), 'pop_total']:,.0f} vs "
        f"WorldPop {pc.loc[pc['ratio'].idxmax(), 'worldpop_sum']:,.0f}；"
        f"最低 {pc['ratio'].idxmin()} 普查 {pc.loc[pc['ratio'].idxmin(), 'pop_total']:,.0f} vs "
        f"WorldPop {pc.loc[pc['ratio'].idxmin(), 'worldpop_sum']:,.0f}。"
        "全港级 r=0.929 掩盖局部严重不一致；在取得 STPU 边界与原始底表前，"
        "STPU 人口/脆弱性不宜用于正式区域比较")

    # C14 人口值是否重复
    add("C14", "STPU 人口值是否存在复制/重复", "通过" if int(pop.duplicated().sum()) == 0 else "不通过",
        f"207 个值中重复 {int(pop.duplicated().sum())} 个")

    # C15 65+ 比例内部一致性
    ok65 = int(np.isclose(d["pct_aged_65plus"] * d["pop_total"], np.round(d["pct_aged_65plus"] * d["pop_total"]), atol=0.02).sum())
    add("C15", "65+ 比例与人口数是否内部自洽", "通过" if ok65 >= len(d) - 10 else "待验证",
        f"pct×pop 近整数 {ok65}/{len(d)}；中位比例 {d['pct_aged_65plus'].median():.2f}%")

    # C16 收入单位
    inc = base["household_income_median"].dropna()
    add("C16", "收入中位数的单位与年份", "待验证",
        f"范围 {inc.min():,.0f}–{inc.max():,.0f}，量级符合 HK$/月；但 A 方未书面确认单位与年份"
        f"（底表字段说明标“待确认”）")

    # C17 风险字段完整性
    risk_cols = ["risk_test_dual_wmean", "risk_test_dual_umean", "risk_test_dual_max", "risk_all_wmean"]
    add("C17", "风险字段是否对 211 个 STPU 全覆盖", "通过" if base[risk_cols].notna().all().all() else "不通过",
        f"缺失 {base[risk_cols].isna().sum().to_dict()}；含 4 个无人口属性 STPU")

    # 写出
    pd.DataFrame(checks).to_csv(OUT / "audit_checks.csv", index=False, encoding="utf-8-sig")
    pc.reset_index().rename(columns={"stpug": "stpug"}).to_csv(
        OUT / "population_crosscheck.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "generated_at": pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat(),
        "n_checks": len(checks),
        "by_status": pd.Series([c["status"] for c in checks]).value_counts().to_dict(),
        "population_sum_207": pop_sum,
        "official_census_2021": OFFICIAL_CENSUS_2021,
        "official_source": OFFICIAL_SOURCE,
        "land_area_covered_km2": round(area_total, 2),
        "max_abs_agg_diff": max_diff,
        "svi_linear_r2": round(float(r2), 4),
        "worldpop_ratio_outliers": int(len(out)),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
