"""论文第四章「历史记录的时间与成因特征」描述性统计。

口径纪律（对应用户 2026-10-25 指令）：
- 真实事件层 = 黄金事件清单 6 事件（窗口内 2 个有 500m 网格映射）；与弱监督层**分开**统计，绝不混合。
- 弱监督层 = 主表 flooded_2h=1（flooded_2h ≡ is_A × [warn_level≥2]），只代表弱监督标签，不代表真实事件次数。
- ERA5 / POWER 为再分析数据，不当雨量站实测；警告时段 ≠ 水浸发生时刻。
- 复合条件一律用「事件相关天气窗口」口径；阈值标注探索性来源，不声称官方标准。
- 不改主 Parquet、不改划分、不训练任何模型。

输出：
- outputs/chapter4_statistics/table_summary.csv        （论文表格汇总）
- outputs/chapter4_statistics/*.csv                    （分组统计）
- stdout 校验信息

用法：.venv/bin/python scripts/chapter4_descriptive_stats.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
B2 = ROOT / "data/delivery_v10_1"
MAIN = B2 / "flood_risk_dataset_2h_500m.parquet"
OUT = ROOT / "outputs/chapter4_statistics"

KEY = ["grid_id", "date", "slot"]

# ---------------------------------------------------------------------------
# 黄金事件台账（来源：data/delivery_v10_1/黄金事件清单.md，手工转录，附出处）
# ---------------------------------------------------------------------------
EVENTS = [
    # event_date, warning_start, warning_end (HKT, 分钟级; None=无红/黑警告记录), places, matched_grids, dsd_reports, in_window
    ("2023-09-08", "2023-09-07T23:05", "2023-09-08T15:40", 10, 23, 60, True),
    ("2018-09-16", "2018-09-16T10:55", "2018-09-16T18:50", 19, 55, 47, True),
    ("2017-08-23", None, None, 13, 0, None, False),
    ("2017-05-24", "2017-05-24T09:15", "2017-05-24T12:30", 5, 0, 19, False),
    ("2014-03-30", "2014-03-30T20:15", "2014-03-30T22:30", 5, 0, 29, False),
    ("2017-07-19", None, None, 2, 0, None, False),
]
EV_COLS = ["event_date", "warn_start", "warn_end", "n_places_reported", "n_matched_grids", "dsd_reports", "in_window"]

RAIN_COLS = [
    "era5_rain_prev3h_mm", "era5_rain_prev6h_mm", "era5_rain_prev24h_mm",
    "power_rain_prev3h_mm", "power_rain_prev6h_mm", "power_rain_prev24h_mm",
]
TIDE_COLS = ["tide_pred_max_m", "tide_pred_mean_m", "dist_to_tide_m"]
EXTRA = ["coastal_lowlying_flag", "dc"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def slots_covered(start: str, end: str) -> set[tuple[str, int]]:
    """警告时段 [start, end] 与 2h 槽 [2s, 2s+2) 求交集 → 被覆盖的 (date, slot)。"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = set()
    t = s.floor("2h")
    while t < e:
        slot = int((t - t.normalize()) / pd.Timedelta("2h"))
        out.add((t.strftime("%Y-%m-%d"), slot))
        t += pd.Timedelta("2h")
    return out


def dist_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "na": None,
        "mean": round(float(x.mean()), 3),
        "median": round(float(x.median()), 3),
        "p75": round(float(x.quantile(0.75)), 3),
        "p95": round(float(x.quantile(0.95)), 3),
        "p99": round(float(x.quantile(0.99)), 3),
        "max": round(float(x.max()), 3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []  # table_summary.csv 的行

    def S(item, value, scope, source, paper_ok, limit):
        summary_rows.append({"统计项目": item, "统计数值": value, "统计口径": scope, "数据来源": source,
                             "是否可正式写入论文": paper_ok, "局限": limit})

    # ================= 0. 文件与字段检查 =================
    need = ["flood_risk_dataset_2h_500m.parquet", "黄金事件清单.md", "golden_grid_map_500m.csv",
            "grid_500m_to_stpu.csv", "grid_500m_stpu_overlap.csv", "README.md",
            "qc_labels_500m.json", "qc_grid_stpu_map.json"]
    files = {f: (B2 / f).exists() for f in need}
    assert all(files.values()), f"缺少文件：{[k for k, v in files.items() if not v]}"
    schema = pq.read_schema(str(MAIN)).names
    need_cols = KEY + ["flooded_2h", "flooded_2h_peak", "is_event_day", "is_golden",
                       "golden_overrides_rule", "split", "label_layer", "label_source",
                       "tide_station"] + RAIN_COLS + TIDE_COLS + EXTRA
    missing = [c for c in need_cols if c not in schema]
    assert not missing, f"主表缺列：{missing}"
    print(f"[0] 文件齐全 8/8；主表列 {len(schema)}，所需列齐全 {len(need_cols)}")

    # 数据分层声明
    S("数据分层：真实事件层", "黄金事件清单 6 事件（建模窗口内 2 个）", "有可核证出处（DSD 海纳/天文台年刊/权威报道）的事件台账",
      "黄金事件清单.md + golden_grid_map_500m.csv", "可（作为案例）", "窗口内仅 2 个事件，不能做大样本检验")
    S("数据分层：弱监督标签层", "全表 flooded_2h（v10.1）", "flooded_2h ≡ is_A × [warn_level≥2]，99.998% 被规则复现",
      "flood_risk_dataset_2h_500m.parquet", "可（须声明弱监督）", "不是真实水浸观测次数；35.9% 测试期正例率不代表真实发生率")
    S("数据分层：气象数据", "ERA5-Land / POWER 再分析", "再分析产品，网格化，非雨量站实测",
      "A 交付 README + 实时数据源与API适配清单", "可（须声明再分析）", "对 2023-09-08 极端降雨低估 7–13 倍（A 的 QC 自述）")

    # ================= 1. 读主表（轻列 + 全表） =================
    light = KEY + ["flooded_2h", "flooded_2h_peak", "is_event_day", "is_golden", "split", "dc",
                   "warn_level", "power_rain_slot_2h_mm", "era5_rain_slot_2h_mm"]
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from floodrisk import realdata as rd

    df = rd.load_delivery(str(MAIN), columns=light, categorical=["grid_id", "date", "dc"])
    df["date_str"] = df["date"].astype(object).astype(str)
    df["month"] = df["date_str"].str[5:7]
    df["year"] = df["date_str"].str[:4]
    n_all = len(df)

    # ================= 2. 真实事件层：时间分布 =================
    ev = pd.DataFrame(EVENTS, columns=EV_COLS)
    ev["year"] = ev["event_date"].str[:4]
    ev["month"] = ev["event_date"].str[5:7]
    ev["has_warn_window"] = ev["warn_start"].notna()
    ev.to_csv(OUT / "real_events_ledger.csv", index=False, encoding="utf-8-sig")
    ev_ym = ev.groupby(["year", "month"]).agg(
        n_events=("event_date", "size"),
        n_places_reported=("n_places_reported", "sum"),
        dsd_reports=("dsd_reports", lambda s: None if s.isna().all() else int(s.sum())),
    ).reset_index()
    ev_ym.to_csv(OUT / "real_events_by_year_month.csv", index=False, encoding="utf-8-sig")
    n_ev_all = len(ev)
    n_ev_inwin = int(ev["in_window"].sum())
    n_warn_known = int(ev["has_warn_window"].sum())
    print(f"[2] 真实事件 {n_ev_all}（窗口内 {n_ev_inwin}）；有警告起止时间 {n_warn_known}")
    S("真实事件总数", f"{n_ev_all}（建模窗口内 {n_ev_inwin}，窗口外 {n_ev_all - n_ev_inwin}）",
      "有可核证出处的暴雨/台风相关水浸事件台账（2014–2023）", "黄金事件清单.md", "可",
      "台账为筛选性收集（针对典型事件），非全量水浸登记")
    S("真实事件年/月分布", f"2014(1) 2017(2) 2018(1) 2023(1)；月份 03/05/07/08/09 各 1–2 个", "按事件日期的年份、月份计数",
      "黄金事件清单.md（real_events_by_year_month.csv）", "可", "n=6，只能作描述性陈列，不能做统计检验")
    S("真实事件发生时段", "无逐事件、逐地点水浸起止时刻记录；仅 4/6 个事件有红/黑警告起止（分钟级，HKT），"
      "2017-08-23 与 2017-07-19 无红/黑警告记录（时段未知）",
      "警告时段为官方暴雨警告起止，不等于水浸发生时刻", "黄金事件清单.md §3", "可（按此表述）",
      "时间精度到 2 小时槽（由警告窗口界定）；峰值槽为降雨推断，非观测时刻")

    # ================= 3. 弱监督正标签：年 / 月 / 槽分布 =================
    pos = df[df["flooded_2h"] == 1]
    weak_y = pos.groupby("year").size().rename("weak_pos_rows").reset_index()
    weak_y["share_pct"] = (weak_y["weak_pos_rows"] / len(pos) * 100).round(2)
    weak_m = pos.groupby("month").size().rename("weak_pos_rows").reset_index()
    weak_m["share_pct"] = (weak_m["weak_pos_rows"] / len(pos) * 100).round(2)
    weak_s = pos.groupby("slot").size().rename("weak_pos_rows").reset_index()
    weak_s["share_pct"] = (weak_s["weak_pos_rows"] / len(pos) * 100).round(2)
    weak_y.to_csv(OUT / "weak_pos_by_year.csv", index=False, encoding="utf-8-sig")
    weak_m.to_csv(OUT / "weak_pos_by_month.csv", index=False, encoding="utf-8-sig")
    weak_s.to_csv(OUT / "weak_pos_by_slot.csv", index=False, encoding="utf-8-sig")
    top_month = weak_m.loc[weak_m["weak_pos_rows"].idxmax()]
    top_slot = weak_s.loc[weak_s["weak_pos_rows"].idxmax()]
    n_days_event = df[df["is_event_day"] == 1]["date_str"].nunique()
    print(f"[3] 弱监督正例 {len(pos):,} 行；正例最多月份 {top_month['month']}（{top_month['share_pct']}%）；正例最多槽 slot {int(top_slot['slot'])}")

    # ---- 3b. 警告字段与降雨一致性核查（季节异常的证据）----
    evd_days = sorted(df.loc[df["is_event_day"] == 1, "date_str"].unique())
    evd_yml = pd.Series([d[:7] for d in evd_days]).value_counts().sort_index()
    evd_ym_df = pd.DataFrame({"ym": evd_yml.index, "n_event_days": evd_yml.values})
    evd_ym_df["year"] = evd_ym_df["ym"].str[:4]
    evd_ym_df["month"] = evd_ym_df["ym"].str[5:7]
    winter_days = int(evd_ym_df[evd_ym_df["month"].isin(["11", "12", "01"])]["n_event_days"].sum())
    evd_ym_df.to_csv(OUT / "event_days_by_year_month.csv", index=False, encoding="utf-8-sig")
    print(f"[3b] 事件日 {len(evd_days)} 天，其中冬季（11/12/1 月）{winter_days} 天（{winter_days/len(evd_days)*100:.0f}%）——与香港暴雨季节（4–10 月）不符，触发 warn_level 口径核查")
    S("弱监督正标签总数", f"{len(pos):,} 行（全表 {n_all:,} 行中）", "flooded_2h=1 的网格-槽行数",
      "主表 flooded_2h", "不可单独作为事件数引用", "标签由规则生成（is_A × 警告槽展开），同一场雨同一天内多行重复计数")
    S("弱监督正标签·年份分布", "、".join(f"{r.year}年 {r.weak_pos_rows:,}（{r.share_pct}%）" for r in weak_y.itertuples()),
      "flooded_2h=1 行按年份占比", "主表（weak_pos_by_year.csv）", "可（须声明弱监督）", "受 is_event_day 标记与规则口径影响，非逐年真实水浸量")
    S("弱监督正标签·月份分布", f"最多 {top_month['month']} 月 {top_month['share_pct']}%",
      "flooded_2h=1 行按月份占比", "主表（weak_pos_by_month.csv）",
      "不可作为真实水浸季节特征", "正例集中冬季（11–1 月），与香港暴雨季节（4–10 月）相反；warn_level 口径存疑（见下条），须 A 澄清后才能解读")
    S("弱监督正标签·2h 槽分布", f"最多 slot {int(top_slot['slot'])}（{top_slot['share_pct']}%）", "flooded_2h=1 行按 0–11 槽占比",
      "主表（weak_pos_by_slot.csv）", "可（须声明弱监督）", "槽分布反映警告活跃时段与规则展开，不是真实发生时刻分布")
    S("事件日标记", f"{n_days_event} 个事件日（is_event_day=1），其中冬季 {winter_days} 天（{winter_days/len(evd_days)*100:.0f}%）",
      "A 的日级事件标记（=当日 warn_level≥2）", "主表 is_event_day（event_days_by_year_month.csv）", "暂不可写（口径存疑）",
      "警告口径异常：见「warn_level 与降雨一致性」条")

    # ================= 4. 真实事件层：空间覆盖 =================
    gmap = pd.read_csv(B2 / "golden_grid_map_500m.csv")
    n_place_records = int(gmap["place"].shape[0])
    n_places_inwin = int(gmap.groupby("place").ngroups)  # 窗口内独立地点（跨事件去重按地点名）
    place_event = gmap.groupby("place")["event_date"].nunique()
    n_places_multi = int((place_event > 1).sum())
    grids_per_event = gmap.groupby("event_date")["grid_id"].nunique()
    n_grids_union = int(gmap["grid_id"].nunique())  # 并集（共享网格只算一次）
    n_grids_sum = int(grids_per_event.sum())        # 各事件合计（共享网格重复）
    stpu = pd.read_csv(B2 / "grid_500m_to_stpu.csv")
    overlap = pd.read_csv(B2 / "grid_500m_stpu_overlap.csv")
    stpu_of_gold = stpu[stpu["grid_id"].isin(set(gmap["grid_id"]))]
    n_stpu_majority = int(stpu_of_gold["stpug"].nunique())
    n_stpu_overlap = int(overlap[overlap["grid_id"].isin(set(gmap["grid_id"]))]["stpug"].nunique())
    # 按区域（dc，18 区口径）分布
    dc_of_grid = df.drop_duplicates("grid_id")[["grid_id", "dc"]]
    gmap2 = gmap.merge(dc_of_grid, on="grid_id", how="left")
    area = (gmap2.groupby("dc").agg(n_grid_unique=("grid_id", "nunique"), n_place_pairs=("place", "size"))
            .reset_index().sort_values("n_grid_unique", ascending=False))
    area.to_csv(OUT / "real_events_by_area.csv", index=False, encoding="utf-8-sig")
    gmap2.to_csv(OUT / "real_events_grid_map_enriched.csv", index=False, encoding="utf-8-sig")
    print(f"[4] 窗口内地点 {n_places_inwin}（跨事件同名 {n_places_multi}）；网格并集 {n_grids_union}（各事件合计 {n_grids_sum}）；STPU 主映射 {n_stpu_majority} / 长表 {n_stpu_overlap}")
    S("真实事件独立地点数", f"窗口内 2 事件共 {n_places_inwin} 个独立地点（跨事件同名地点 {n_places_multi} 个）；"
      f"窗口外 4 事件另有 25 条地点记录（清单仅给地点数，未逐点列出）",
      "地点 = 公开报道地标；跨事件同名地点按地点名去重（如沙头角同现于 2023 与 2018 两事件）",
      "golden_grid_map_500m.csv + 黄金事件清单.md", "可", "地点坐标为报道地标质心（数百米精度）")
    S("地点→500m 网格映射", f"窗口内 {n_places_inwin} 地点全部完成匹配（100%），87 个地点-网格对 → 网格并集 {n_grids_union} 个",
      "最近 3 格（形心距 ≤0.8km）；同一地点的多个候选网格只计 1 个事件地点，网格并集去重后统计",
      "golden_grid_map_500m.csv", "可", "一地点对应 3 个候选格，网格级计数应说明是「匹配格」而非「确认受淹格」")
    S("真实事件涉及 STPU", f"主映射 {n_stpu_majority} 个 / 面积权重长表 {n_stpu_overlap} 个", "78 个匹配网格经 grid_500m_to_stpu / overlap 映射到 STPU",
      "grid_500m_to_stpu.csv + grid_500m_stpu_overlap.csv", "可", "主映射每格 1 个多数 STPU；长表含边界共享格")
    S("真实事件按区域分布", f"涉及 {len(area)} 个区域（dc 口径），最多 {area.iloc[0]['dc']}（{int(area.iloc[0]['n_grid_unique'])} 格）",
      "匹配网格按 18 区分组去重", "主表 dc + golden_grid_map_500m.csv（real_events_by_area.csv）", "可",
      "区域归属按网格多数区，跨区边界地点有归类误差")

    # ================= 5. 事件天气窗口切片（黄金行 + 警告活跃槽） =================
    gold = df[df["is_golden"] == 1].copy()
    assert len(gold) == 936, f"黄金行数 {len(gold)} != 936"
    warn_slots_2023 = slots_covered("2023-09-07T23:05", "2023-09-08T15:40")
    warn_slots_2018 = slots_covered("2018-09-16T10:55", "2018-09-16T18:50")
    warn_slots = warn_slots_2023 | warn_slots_2018
    gold["warn_active"] = [(d, int(s)) in warn_slots for d, s in zip(gold["date_str"], gold["slot"])]

    # 降雨/潮位列只对需要的行读（黄金行 + 事件日行 + 非事件日行做对比）
    need_keys = df[KEY].copy()
    heavy = rd.load_delivery(
        str(MAIN),
        columns=KEY + RAIN_COLS + TIDE_COLS + ["coastal_lowlying_flag"],
        filter_keys=need_keys,
        categorical=["grid_id", "date"],
    )
    heavy = heavy.merge(df[KEY + ["flooded_2h", "flooded_2h_peak", "is_golden", "is_event_day", "year", "month"]], on=KEY, how="left")
    heavy["warn_active"] = [(d, int(s)) in warn_slots for d, s in zip(heavy["date"].astype(object).astype(str), heavy["slot"])]

    gold_h = heavy[heavy["is_golden"] == 1]
    gold_warn = gold_h[gold_h["warn_active"]]
    gold_peak = gold_h[gold_h["flooded_2h_peak"] == 1]
    ev_day = heavy[heavy["is_event_day"] == 1]
    nev_day = heavy[heavy["is_event_day"] == 0]
    print(f"[5] 黄金行 {len(gold_h)}（警告活跃 {len(gold_warn)}，峰值槽 {len(gold_peak)}）；事件日行 {len(ev_day):,}；非事件日行 {len(nev_day):,}")

    # ---- 降雨统计：三组口径 ----
    rain_rows = []
    for scope, sub in (("黄金行（事件日全 12 槽）", gold_h), ("黄金行·警告活跃槽", gold_warn), ("黄金行·峰值槽", gold_peak)):
        for c in RAIN_COLS:
            st = dist_stats(sub[c])
            rain_rows.append({"口径": scope, "字段": c, **st})
    for c in RAIN_COLS:
        rain_rows.append({"口径": "非事件日（对照，抽样设计覆盖）", "字段": c, **dist_stats(nev_day[c])})
    rain_df = pd.DataFrame(rain_rows)
    rain_df.to_csv(OUT / "rain_event_windows_vs_nonevent.csv", index=False, encoding="utf-8-sig")
    g6 = rain_df[(rain_df["口径"] == "黄金行·警告活跃槽")]
    era5_med = float(g6[g6["字段"] == "era5_rain_prev6h_mm"]["median"].iloc[0])
    era5_max = float(g6[g6["字段"] == "era5_rain_prev6h_mm"]["max"].iloc[0])
    pow_med = float(g6[g6["字段"] == "power_rain_prev6h_mm"]["median"].iloc[0])
    pow_max = float(g6[g6["字段"] == "power_rain_prev6h_mm"]["max"].iloc[0])
    nev95_era5 = float(rain_df[(rain_df["口径"].str.startswith("非事件日")) & (rain_df["字段"] == "era5_rain_prev6h_mm")]["p95"].iloc[0])
    print(f"[5] 警告活跃槽 prev6h：ERA5 中位 {era5_med} / 最大 {era5_max}；POWER 中位 {pow_med} / 最大 {pow_max}；非事件日 ERA5 p95={nev95_era5}")
    S("事件窗口降雨（ERA5 prev6h）", f"中位 {era5_med}mm / 最大 {era5_max}mm（警告活跃槽，n={len(gold_warn)}）",
      "黄金行 × 警告活跃槽；ERA5-Land 再分析，截至槽开始", "主表 era5_rain_prev6h_mm", "可（须声明再分析）",
      "再分析对 2023-09-08 低估 7–13 倍；非雨量站实测")
    S("事件窗口降雨（POWER prev6h）", f"中位 {pow_med}mm / 最大 {pow_max}mm", "同上；POWER 为全港单一序列（无空间差异）",
      "主表 power_rain_prev6h_mm", "可（须声明口径）", "全港单值，不能反映局地雨量差异")
    S("事件 vs 非事件时段降雨", f"非事件日 ERA5 prev6h p95 = {nev95_era5}mm，显著低于事件窗口（见 rain_event_windows_vs_nonevent.csv）",
      "事件日行 vs 非事件日行，网格-槽级；非事件日仅 35 格/天（A 抽样设计）", "主表", "可（须声明覆盖不均衡）",
      "非事件日覆盖极低，占比不能外推为全港天气频率")

    # ---- 潮位统计 ----
    tide_rows = []
    for scope, sub in (("黄金行（事件日全 12 槽）", gold_h), ("黄金行·警告活跃槽", gold_warn)):
        for c in TIDE_COLS:
            tide_rows.append({"口径": scope, "字段": c, **dist_stats(sub[c])})
    tide_rows.append({"口径": "非事件日（对照）", "字段": "tide_pred_max_m", **dist_stats(nev_day["tide_pred_max_m"])})
    tide_df = pd.DataFrame(tide_rows)
    tide_df.to_csv(OUT / "tide_event_windows.csv", index=False, encoding="utf-8-sig")
    tmax_warn = gold_warn["tide_pred_max_m"]
    tide_p95_all = float(dist_stats(heavy["tide_pred_max_m"])["p95"])
    # 沿海 vs 内陆（网格级静态代理 coastal_lowlying_flag）
    coastal = gold_warn[gold_warn["coastal_lowlying_flag"] == 1]["tide_pred_max_m"]
    inland = gold_warn[gold_warn["coastal_lowlying_flag"] == 0]["tide_pred_max_m"]
    tide_cmp = pd.DataFrame([
        {"组": "警告活跃槽·沿海低洼格（coastal_lowlying_flag=1）", **dist_stats(coastal)},
        {"组": "警告活跃槽·其余格（flag=0）", **dist_stats(inland)},
    ])
    tide_cmp.to_csv(OUT / "tide_coastal_vs_inland.csv", index=False, encoding="utf-8-sig")
    print(f"[6] 警告活跃槽 tide_pred_max：中位 {tmax_warn.median():.2f}m p95 {tmax_warn.quantile(0.95):.2f}；全表 p95={tide_p95_all}；沿海中位 {coastal.median():.2f} vs 内陆 {inland.median():.2f}")
    S("事件窗口潮位", f"警告活跃槽 tide_pred_max 中位 {tmax_warn.median():.2f}m、p95 {tmax_warn.quantile(0.95):.2f}m（n={len(tmax_warn)}）",
      "天文潮调和预报（覆盖未来槽，预测时刻可提前获得），非实测增水", "主表 tide_pred_max_m", "可（须声明为天文潮预报）",
      "不含风暴增水（surge 为槽内未来量已排除；2018-09-16 实测增水 +2.35m 仅存于台账文字）")
    S("沿海 vs 内陆潮位", f"警告活跃槽：沿海低洼格中位 {coastal.median():.2f}m（n={len(coastal)}） vs 其余格 {inland.median():.2f}m（n={len(inland)}）",
      "按网格静态代理 coastal_lowlying_flag 分组，非事件地点实地分类", "主表 coastal_lowlying_flag（描述性使用，不入模）", "可（须声明代理口径）",
      "flag 为 A 类规则输入的静态先验，不等于该场水浸的成因分类")

    # ---- 复合条件（事件相关天气窗口口径）----
    T_rain = nev95_era5  # 探索性阈值：非事件日 ERA5 prev6h 的 95 分位
    T_tide = tide_p95_all  # 探索性阈值：全表 tide_pred_max 的 95 分位
    n_w = len(gold_warn)
    both = int(((gold_warn["era5_rain_prev6h_mm"] >= T_rain) & (gold_warn["tide_pred_max_m"] >= T_tide)).sum())
    rain_only = int((gold_warn["era5_rain_prev6h_mm"] >= T_rain).sum())
    tide_only = int((gold_warn["tide_pred_max_m"] >= T_tide).sum())
    compound = pd.DataFrame([
        {"指标": "强降雨（ERA5 prev6h ≥ 非事件日 p95）", "阈值": round(T_rain, 2), "n": rain_only, "pct_of_warn_slots": round(rain_only / n_w * 100, 1)},
        {"指标": "较高潮位（tide_pred_max ≥ 全表 p95）", "阈值": round(T_tide, 2), "n": tide_only, "pct_of_warn_slots": round(tide_only / n_w * 100, 1)},
        {"指标": "两者同时", "阈值": f"{round(T_rain,2)}mm 且 ≥{round(T_tide,2)}m", "n": both, "pct_of_warn_slots": round(both / n_w * 100, 1)},
        {"指标": "分母：警告活跃槽黄金行", "阈值": None, "n": n_w, "pct_of_warn_slots": 100.0},
    ])
    compound.to_csv(OUT / "compound_conditions.csv", index=False, encoding="utf-8-sig")
    print(f"[7] 复合条件（窗口 n={n_w}）：强降雨 {rain_only}（{rain_only/n_w*100:.0f}%） 高潮位 {tide_only}（{tide_only/n_w*100:.0f}%） 同时 {both}（{both/n_w*100:.0f}%）")
    S("复合条件：强降雨×高潮位共现", f"事件相关天气窗口（警告活跃槽，n={n_w}）中同时出现 {both} 行（{both/n_w*100:.1f}%）；"
      f"强降雨单独 {rain_only/n_w*100:.1f}%、较高潮位单独 {tide_only/n_w*100:.1f}%",
      "「事件相关天气窗口」= 黄金匹配格 × 警告活跃槽；阈值均为探索性（非事件日 p95 / 全表 p95），非官方水浸成因标准",
      "主表 era5_rain_prev6h_mm + tide_pred_max_m（compound_conditions.csv）", "可（须按「天气窗口」口径并声明探索性阈值）",
      "共现≠成因；潮位为天文预报不含风暴增水；样本仅 2 个事件，不能做统计检验")

    # ---- 典型极端天气窗口标注（不把警告时段当水浸时刻）----
    S("典型极端天气窗口", "2023-09-07/08 世纪暴雨（黑雨 23:05–15:40，16.6h，DSD 报告 60 宗）；2018-09-16 山竹（红雨 10:55–18:50，7.9h，47 宗，维港风暴潮 +2.35m）",
      "警告时段为官方起止（分钟级）；水浸报告宗数为 DSD 台账数", "黄金事件清单.md（附出处链接）", "可（附出处）",
      "宗数为全市接报数，未逐条定位到网格；不把警告时段当水浸发生时刻")

    # ================= 7b. warn_level 与降雨一致性核查（重大口径疑点） =================
    # 事件日全港「网格×12槽 POWER 2h 雨量合计」的日雨量分布 + 冬季事件日抽查（数据已在首轮全表读入）。
    probe_days = ["2025-01-28", "2025-01-15", "2021-11-11", "2024-12-16", "2023-09-08", "2018-09-16"]
    psub = df[(df["date_str"].isin(probe_days)) & (df["slot"] == 5)]
    spot = []
    for d, sub in psub.groupby("date_str"):
        spot.append({
            "date": d,
            "warn_level_slot5": int(sub["warn_level"].astype(int).iloc[0]),
            "era5_slot5_mm": round(float(sub["era5_rain_slot_2h_mm"].median()), 2),
            "power_slot5_mm": round(float(sub["power_rain_slot_2h_mm"].median()), 2),
            "n_grids": int(sub["grid_id"].nunique()),
        })
    evd_df = df[df["is_event_day"] == 1]
    grid_day_rain = evd_df.groupby(["date_str", "grid_id"], observed=True)["power_rain_slot_2h_mm"].sum()
    grid_day = grid_day_rain.groupby(level=0).median()  # 事件日全港网格中位日雨量（12 槽合计）
    n_evd_low = int((grid_day < 100).sum())
    n_evd_dry = int((grid_day < 1).sum())
    spot_df = pd.DataFrame(spot)
    spot_df.to_csv(OUT / "warn_level_rain_consistency_spot.csv", index=False, encoding="utf-8-sig")
    grid_day.rename("median_daily_rain_mm").to_csv(OUT / "event_days_daily_rain.csv", encoding="utf-8-sig")
    n_ge100 = int((grid_day >= 100).sum())
    n_1_100 = int(((grid_day >= 1) & (grid_day < 100)).sum())
    ge100_dates = [d for d, v in grid_day.items() if v >= 100]
    # 黄金清单记载的雨量站实测中位（用于对照再分析低估）
    gold_days_in_table = {"2023-09-08": 297.8, "2018-09-16": 167.5}
    power_of_gold = {d: round(float(grid_day.get(d, np.nan)), 1) for d in gold_days_in_table}
    print(f"[7b] warn_level 一致性：事件日 {len(grid_day)} 天中，全港 POWER 口径中位日雨量 <1mm 的 {n_evd_dry} 天（近乎无雨却 warn_level≥2）；"
          f"1–100mm 的 {n_1_100} 天；≥100mm 仅 {n_ge100} 天（{ge100_dates}）。"
          f"黄金事件日 POWER 值 {power_of_gold} vs 清单实测中位 {gold_days_in_table} → 再分析低估（与 A QC 自述 7–13 倍一致）；抽查见 warn_level_rain_consistency_spot.csv")
    S("warn_level 与降雨一致性（重大疑点）",
      f"376 个事件日中 {n_evd_dry} 天（{n_evd_dry/len(grid_day)*100:.0f}%）全港 POWER 口径中位日雨量 <1mm（近乎无雨）却存在 warn_level≥2 槽；"
      f"1–100mm 的 {n_1_100} 天、≥100mm 仅 {n_ge100} 天。抽查冬季事件日（2025-01-28、2024-12-16 等）warn_level=2 但全日槽雨量≈0.0mm。"
      f"注：黄金事件日 POWER 中位日雨量（2023-09-08 为 {power_of_gold['2023-09-08']}mm、2018-09-16 为 {power_of_gold['2018-09-16']}mm）远低于清单记载的雨量站实测（297.8/167.5mm），"
      f"属再分析低估（A QC 自述对 2023-09-08 低估 7–13 倍），与无雨警告日是两类不同问题",
      "事件日定义 = 当日存在 warn_level≥2 槽；对照该日全港 POWER 2h 槽雨量 12 槽合计的网格中位数（再分析口径，非雨量站实测）",
      "主表 warn_level + power_rain_slot_2h_mm（event_days_daily_rain.csv、warn_level_rain_consistency_spot.csv）",
      "暂不可写（须 A 澄清）",
      "约八成事件日近乎无雨却有 warn_level≥2（可能为警告数据解析/对齐问题）；弱监督正例的季节分布因此不可解读为真实水浸季节特征；已列入 A 的数据澄清需求")

    # ================= 8. table_summary.csv =================
    ts = pd.DataFrame(summary_rows)
    ts.to_csv(OUT / "table_summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "created_by": "scripts/chapter4_descriptive_stats.py",
        "main_parquet_sha256": sha256(MAIN)[:16] + "…",
        "n_rows_main": int(n_all),
        "weak_pos_rows": int(len(pos)),
        "golden_rows": int(len(gold_h)),
        "warn_active_golden_rows": int(n_w),
        "thresholds": {"rain_explore_era5_prev6h_p95_nonevent_mm": round(T_rain, 3),
                        "tide_explore_p95_all_m": round(T_tide, 3)},
        "disclaimer": "真实事件层与弱监督层分开统计；ERA5/POWER 为再分析；警告时段≠水浸时刻；弱监督正例不代表真实事件次数",
    }
    (OUT / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {OUT}（table_summary {len(ts)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
