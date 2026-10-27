"""论文第四章四张表格：从现有数据与实验产物提取可确认的统计（不训练、不重读主 Parquet）。

数据来源（全部只读）：
- data/delivery_v10_1/golden_grid_map_500m.csv        窗口内事件地点-网格匹配（87 行）
- data/delivery_v10_1/grid_500m_to_stpu.csv           网格→STPU 主映射 + STPU 级社区属性（5,517 行）
- data/delivery_v10_1/grid_500m_stpu_overlap.csv      网格→STPU 面积权重长表（7,703 行）
- data/delivery_v10_1/黄金事件清单.md                 6 事件台账（手工转录为 EVENTS 常量，附出处）
- outputs/chapter4_statistics/*.csv                   上一脚本产出的时间分布与降雨/潮位/复合统计（直接复用）
- outputs/small_experiment_v10_1_300k_20261024-150953/metrics.json   300k 消融实验指标

口径纪律（对应用户 2026-10-25 四表任务书）：
- 真实事件 / 黄金事件 / 历史黑点 / 弱监督标签 / 网格时槽样本五类分开，绝不混合。
- STPU 人口等社区属性为 STPU 级字段，不从网格重分配、不把 STPU 人口重复摊给多格。
- exposure_population 只称「潜在暴露人口（估算）」，不称实际受灾人数。
- 0.30 为参考阈值；实验指标只列「探索性实验参考」，不进正式验证表。
- 缺证据字段一律「待核验 / 无法统计 / 待正式模型测试」，不编造。

输出：outputs/chapter4_tables/table{1..4}_*.csv

用法：.venv/bin/python scripts/chapter4_extract_existing_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B2 = ROOT / "data/delivery_v10_1"
PREV = ROOT / "outputs/chapter4_statistics"          # 上一脚本（chapter4_descriptive_stats.py）产物
EXP = ROOT / "outputs/small_experiment_v10_1_300k_20261024-150953"
OUT = ROOT / "outputs/chapter4_tables"

# 黄金事件台账（与 scripts/chapter4_descriptive_stats.py 同一口径，来源：黄金事件清单.md）
EVENTS = [
    ("2023-09-08", "黑色暴雨警告", "2023-09-07T23:05", "2023-09-08T15:40", 10, 23, 60, "DSD《海纳》无此年宗数；官方/权威报道", True),
    ("2018-09-16", "红色暴雨警告（台风山竹）", "2018-09-16T10:55", "2018-09-16T18:50", 19, 55, 47, "DSD《海纳 2009-2018》+ 天文台《2018 热带气旋年刊》", True),
    ("2017-08-23", "台风天鸽（十号风球，恰逢天文大潮）", None, None, 13, 0, None, "DSD《海纳 2009-2018》+ 天文台《2017 热带气旋年刊》", False),
    ("2017-05-24", "黑色暴雨警告", "2017-05-24T09:15", "2017-05-24T12:30", 5, 0, 19, "DSD《海纳 2009-2018》", False),
    ("2014-03-30", "黑色暴雨警告伴冰雹", "2014-03-30T20:15", "2014-03-30T22:30", 5, 0, 29, "DSD《海纳 2009-2018》", False),
    ("2017-07-19", "暴雨（无红/黑警告记录）", None, None, 2, 0, None, "DSD《海纳 2009-2018》", False),
]
EV_COLS = ["event_date", "event_type", "warn_start", "warn_end", "n_places_reported",
           "n_matched_grids", "dsd_reports", "record_source", "in_window"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ================= 表 1：历史记录的时间与成因特征 =================
    ev = pd.DataFrame(EVENTS, columns=EV_COLS)
    ev["has_warn_time"] = np.where(ev["warn_start"].notna(), "有（分钟级警告起止）", "未知（无红/黑警告记录）")
    ev["occurrence_time_note"] = np.where(
        ev["warn_start"].notna(),
        "仅警告时段（官方起止，≠水浸发生时刻）；槽级归因至 2h 槽，峰值槽为降雨推断",
        "未知",
    )
    ev.to_csv(OUT / "table1_real_events_ledger.csv", index=False, encoding="utf-8-sig")

    ev["year"] = ev["event_date"].str[:4]
    ev["month"] = ev["event_date"].str[5:7]
    ev[["year", "month", "event_date", "event_type", "has_warn_time", "n_places_reported", "dsd_reports", "in_window"]].to_csv(
        OUT / "table1_time_real_events.csv", index=False, encoding="utf-8-sig")

    # 弱监督标签分布：复用上一脚本产物，三段合并并明确标注「非真实水浸次数」
    parts = []
    for dim, f in (("year", "weak_pos_by_year.csv"), ("month", "weak_pos_by_month.csv"), ("slot_2h", "weak_pos_by_slot.csv")):
        d = pd.read_csv(PREV / f)
        d.columns = [dim, "weak_pos_rows", "share_pct"]
        d.insert(0, "维度", dim)
        parts.append(d)
    weak = pd.concat(parts, ignore_index=True)
    weak["口径"] = "flooded_2h=1 网格-槽行数（弱监督标签，规则展开，同一场雨多行重复计数；非真实水浸次数）"
    weak["数据来源"] = "delivery_v10_1 主表 flooded_2h（由 outputs/chapter4_statistics/ 复用）"
    weak.to_csv(OUT / "table1_time_weak_labels.csv", index=False, encoding="utf-8-sig")

    # 空间覆盖（从 gmap 现算，口径与上一脚本一致）
    gmap = pd.read_csv(B2 / "golden_grid_map_500m.csv")
    stpu = pd.read_csv(B2 / "grid_500m_to_stpu.csv")
    overlap = pd.read_csv(B2 / "grid_500m_stpu_overlap.csv")
    dc_of_grid = pd.read_csv(PREV / "real_events_grid_map_enriched.csv")  # 已含 dc（上一脚本合并产物）
    grids_union = set(gmap["grid_id"])
    stpu_major = stpu[stpu["grid_id"].isin(grids_union)]["stpug"].nunique()
    stpu_overlap_n = overlap[overlap["grid_id"].isin(grids_union)]["stpug"].nunique()
    gmap2 = gmap.merge(stpu[["grid_id", "stpug", "stpug_chi"]], on="grid_id", how="left").merge(
        dc_of_grid[["grid_id", "dc"]].drop_duplicates(), on="grid_id", how="left")
    gmap2.to_csv(OUT / "table1_event_places_grid_stpu.csv", index=False, encoding="utf-8-sig")
    spatial = pd.DataFrame([
        {"统计项": "真实事件台账数", "数值": len(ev), "口径": "有可核证出处的典型事件（2014–2023），筛选性收集"},
        {"统计项": "独立事件数（按事件日期去重）", "数值": ev["event_date"].nunique(), "口径": "6 条台账对应 6 个不同日期，无重复"},
        {"统计项": "建模窗口内事件", "数值": int(ev["in_window"].sum()), "口径": "2018-01-01–2026-09-30 内：2023-09-08、2018-09-16"},
        {"统计项": "可定位独立地点（窗口内 2 事件）", "数值": int(gmap["place"].nunique()), "口径": "报道地标去重（沙头角跨事件同名计 1）"},
        {"统计项": "地点-网格匹配对", "数值": len(gmap), "口径": "最近 3 格、形心距 ≤0.8km；匹配成功率 100%"},
        {"统计项": "涉及 500m 网格（并集）", "数值": len(grids_union), "口径": "各事件合计 78，跨事件共享 3 格去重"},
        {"统计项": "涉及 STPU（主映射/面积长表）", "数值": f"{stpu_major} / {stpu_overlap_n}", "口径": "75 格并集 → grid_500m_to_stpu（每格 1 个多数 STPU）/ overlap 长表"},
        {"统计项": "地点级时槽样本（黄金行）", "数值": 936, "口径": "78 格日对 × 12 槽（L2_golden 层）；警告活跃槽 459、峰值槽 78"},
    ])
    spatial.to_csv(OUT / "table1_spatial_coverage.csv", index=False, encoding="utf-8-sig")

    # 记录来源类别
    sources = pd.DataFrame([
        {"来源类别": "政府部门记录（渠务署 DSD）", "事件数": 4, "说明": "水浸报告宗数：2023-09-08=60、2018-09-16=47、2017-05-24=19、2014-03-30=29；2017-08-23、2017-07-19 未记录"},
        {"来源类别": "官方气象记录（天文台）", "事件数": 4, "说明": "红/黑暴雨警告起止（分钟级）4 个；热带气旋年刊 2 个（2017 天鸽、2018 山竹）"},
        {"来源类别": "历史黑点资料（DSD 水浸黑点名单）", "事件数": None, "说明": "无法统计：交付与项目内无黑点名单数据"},
        {"来源类别": "公众报告（媒体报道）", "事件数": 1, "说明": "2023-09-08 世纪暴雨权威报道（清单附 2 条链接）；地点列表来源之一"},
        {"来源类别": "其他", "事件数": None, "说明": "无法统计：无其他原始来源"},
    ])
    sources["口径"] = "按清单标注出处逐事件归类；来源类别不解释为水浸成因"
    sources.to_csv(OUT / "table1_record_sources.csv", index=False, encoding="utf-8-sig")

    # 降雨 / 潮位 / 复合条件：复用上一脚本产物
    for src, dst in (("rain_event_windows_vs_nonevent.csv", "table1_rain_windows.csv"),
                     ("tide_event_windows.csv", "table1_tide_windows.csv"),
                     ("compound_conditions.csv", "table1_compound_conditions.csv")):
        shutil.copy(PREV / src, OUT / dst)

    # ================= 表 2：模型验证设计与指标结果 =================
    m = json.load(open(EXP / "metrics.json"))
    ab = m["step8_ablation"]
    rows = []
    for fs in ("power_only", "dual"):
        for model in ("xgb", "rf", "lr"):
            t = ab[fs][model]["test"]["overall"]
            v = ab[fs][model]["val"]
            rows.append({
                "特征方案": fs, "模型": model.upper(),
                "ROC_AUC_test": t["roc_auc"], "PR_AUC_test": t["pr_auc"], "Brier_test": t["brier"],
                "Recall_test": t["recall"], "Precision_test": t["precision"],
                "ROC_AUC_val": v["roc_auc"], "PR_AUC_val": v["pr_auc"], "Brier_val": v["brier"],
                "Recall_val": v["recall"], "Precision_val": v["precision"],
                "阈值": t["threshold"], "测试集n": t["n"], "测试集正例率": t["positive_rate"],
            })
    expl = pd.DataFrame(rows)
    expl["实验版本"] = "v10.1（batch 2，SHA256 9f580846…）300k 分层抽样实验"
    expl["样本规模"] = "300,001（train 2018–2021 / val 2022 / test 2023-01–2026-09 时间外，seed=42）"
    expl["标签类型"] = "flooded_2h 弱监督标签（规则生成，非真实水浸观测）"
    expl["性质"] = "探索性实验参考——不得填入正式验证结果表；0.30 为参考阈值，非最佳阈值"
    expl["来源文件"] = "outputs/small_experiment_v10_1_300k_20261024-150953/metrics.json（step8_ablation）"
    expl["备注"] = "power_only=51 列 POWER 特征；dual=59 列（+8 ERA5）；另 static_only/rain_time_only/rain_time_power 见同文件"
    expl.to_csv(OUT / "table2_metrics_exploratory.csv", index=False, encoding="utf-8-sig")

    formal = pd.DataFrame([{
        "模型": "XGBoost（候选主模型）", "训练集年份": "待正式训练", "验证集年份": "待正式训练",
        "测试集年份": "待正式训练", "空间分组验证": "待正式测试",
        "ROC_AUC": "", "PR_AUC": "", "Brier": "", "Recall@阈值": "", "Precision@阈值": "",
        "正样本比例": "", "状态": "待正式模型测试",
    }, {
        "模型": "Random Forest（对照）", "训练集年份": "待正式训练", "验证集年份": "待正式训练",
        "测试集年份": "待正式训练", "空间分组验证": "待正式测试",
        "ROC_AUC": "", "PR_AUC": "", "Brier": "", "Recall@阈值": "", "Precision@阈值": "",
        "正样本比例": "", "状态": "待正式模型测试",
    }, {
        "模型": "Logistic Regression（对照）", "训练集年份": "待正式训练", "验证集年份": "待正式训练",
        "测试集年份": "待正式训练", "空间分组验证": "待正式测试",
        "ROC_AUC": "", "PR_AUC": "", "Brier": "", "Recall@阈值": "", "Precision@阈值": "",
        "正样本比例": "", "状态": "待正式模型测试",
    }])
    formal.to_csv(OUT / "table2_formal_validation_structure.csv", index=False, encoding="utf-8-sig")

    split_dates = m["step3_split"]["info"]["date_ranges"]
    split_counts = m["step3_split"]["info"]["counts"]

    # ================= 表 3：社区风险与脆弱性（STPU 层，属性不从网格重分配） =================
    # 一致性检查：同 STPU 的属性是否为常量（决定能否直接按 STPU 汇总）
    attr_cols = ["pop_total", "pop_density_km2", "pct_aged_65plus", "household_income_median",
                 "social_vulnerability_index", "vulnerability_rank", "exposure_population"]
    n_inconsistent = {c: int((stpu.groupby("stpug")[c].nunique() > 1).sum()) for c in attr_cols}
    stpu_prof = stpu.groupby("stpug", as_index=False).agg(
        stpug_chi=("stpug_chi", "first"),
        n_grids_primary=("grid_id", "size"),
        pop_total=("pop_total", "first"),
        pop_density_km2=("pop_density_km2", "first"),
        pct_aged_65plus=("pct_aged_65plus", "first"),
        household_income_median=("household_income_median", "first"),
        social_vulnerability_index=("social_vulnerability_index", "first"),
        vulnerability_rank=("vulnerability_rank", "first"),
        exposure_population=("exposure_population", "first"),
    )
    ovl_per_stpu = overlap.groupby("stpug")["grid_id"].nunique().rename("n_grids_any_overlap")
    stpu_prof = stpu_prof.merge(ovl_per_stpu, on="stpug", how="left")
    stpu_prof["n_grids_any_overlap"] = stpu_prof["n_grids_any_overlap"].fillna(0).astype(int)
    # 脆弱性分位（211 个 STPU 上的探索性五分位；方向核查：svi 越大 rank 越小 = rank 1 最脆弱）
    spearman = float(stpu_prof[["social_vulnerability_index", "vulnerability_rank"]].corr(method="spearman").iloc[0, 1])
    stpu_prof["svi_quintile_explore"] = pd.qcut(
        stpu_prof["social_vulnerability_index"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"])
    stpu_prof.to_csv(OUT / "table3_stpu_profile.csv", index=False, encoding="utf-8-sig")

    ev_stpu_ids = set(stpu[stpu["grid_id"].isin(grids_union)]["stpug"])
    prof_main = stpu_prof[stpu_prof["stpug"].isin(ev_stpu_ids)]
    summ = pd.DataFrame([
        {"组": "全港（主映射覆盖的 STPU）", "n_STPU": int(stpu_prof["stpug"].nunique()),
         "STPU人口合计(pop_total加总)": round(float(stpu_prof["pop_total"].sum())),
         "65岁以上比例中位(%)": round(float(stpu_prof["pct_aged_65plus"].median()), 2),
         "SVI中位": round(float(stpu_prof["social_vulnerability_index"].median()), 3),
         "潜在暴露人口合计(exposure_population加总)": round(float(stpu_prof["exposure_population"].sum()))},
        {"组": "事件涉及（75 格并集 → 主映射 STPU）", "n_STPU": int(len(prof_main)),
         "STPU人口合计(pop_total加总)": round(float(prof_main["pop_total"].sum())),
         "65岁以上比例中位(%)": round(float(prof_main["pct_aged_65plus"].median()), 2),
         "SVI中位": round(float(prof_main["social_vulnerability_index"].median()), 3),
         "潜在暴露人口合计(exposure_population加总)": round(float(prof_main["exposure_population"].sum()))},
    ])
    summ["口径"] = ("STPU 级字段直接汇总（不从网格重分配）；人口为 STPU 层估算，加总仅为口径展示，"
                    "事件组「涉及社区的属性」≠ 受灾人口；潜在暴露人口为估算值，非实际受灾人数")
    summ["数据来源"] = "grid_500m_to_stpu.csv + golden_grid_map_500m.csv"
    summ.to_csv(OUT / "table3_stpu_summary.csv", index=False, encoding="utf-8-sig")

    ev_stpu = prof_main.copy()
    ev_stpu["涉及事件"] = ev_stpu["stpug"].map(
        lambda s: "、".join(sorted(gmap2.loc[gmap2["stpug"] == s, "event_date"].astype(str).unique())))
    ev_stpu.to_csv(OUT / "table3_event_stpu_profile.csv", index=False, encoding="utf-8-sig")

    # ================= 表 4：历史黑点治理优先序（仅历史证据，不排序） =================
    places = gmap2.groupby(["event_date", "place"], as_index=False).agg(
        n_matched_grids=("grid_id", "nunique"),
        grid_ids=("grid_id", lambda s: "、".join(sorted(set(s)))),
        stpu_primary=("stpug", lambda s: "、".join(sorted(set(s.dropna())))),
        dc=("dc", "first"),
    )
    prof_by_stpu = stpu_prof.set_index("stpug")
    for col, name in (("pop_total", "社区人口(STPU级)"), ("pct_aged_65plus", "65岁以上比例%(STPU级)"),
                      ("social_vulnerability_index", "SVI(STPU级)"), ("exposure_population", "潜在暴露人口_估算_STPU级")):
        places[name] = places["stpu_primary"].map(lambda s: prof_by_stpu.loc[s.split("、")[0], col] if s else np.nan)
    places["已核实历史记录及年份"] = places["event_date"].map(ev.set_index("event_date")["record_source"]) \
        + "；" + places["event_date"].map(lambda d: f"DSD水浸报告宗数={ev.set_index('event_date').loc[d, 'dsd_reports']}"
                                          if pd.notna(ev.set_index('event_date').loc[d, 'dsd_reports']) else "DSD宗数未记录")
    places["排水设施/工程/清淤状态"] = "无法统计（现有资料不含，无来源日期）"
    places["建议核验事项"] = "待核验（现场受淹范围、排水设施现状；不得由历史记录推断最新工程状态）"
    places["预测风险"] = "待正式模型输出"
    places["优先序"] = "留待后续风险与脆弱性综合分析（本次不生成排名）"
    places["历史证据性质"] = "黄金事件匹配格（最近 3 格候选，非确认受淹格）；坐标为报道地标质心"
    places.to_csv(OUT / "table4_places_history.csv", index=False, encoding="utf-8-sig")

    # ================= 控制台校验输出 =================
    print(f"[表1] 台账 {len(ev)}；地点 {gmap['place'].nunique()}（{len(gmap)} 对）；网格并集 {len(grids_union)}；STPU {stpu_major}/{stpu_overlap_n}")
    print(f"[表2] 指标行 {len(expl)}（power_only/dual × 3 模型）；实验切分 {split_dates}；行数 {split_counts}")
    print(f"[表3] STPU 主映射覆盖 {stpu_prof['stpug'].nunique()}（全港 211）；属性跨格不一致的 STPU 数 {n_inconsistent}")
    print(f"[表3] SVI 与 rank 的 Spearman = {spearman:.3f}（正=rank 大更脆弱，负=rank 1 最脆弱）")
    print(f"[表3] exposure_population 与 pop_total 恒等: {bool((stpu['exposure_population'] == stpu['pop_total']).all())}")
    print(f"[表4] 地点-事件行 {len(places)}")
    print(f"[done] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
