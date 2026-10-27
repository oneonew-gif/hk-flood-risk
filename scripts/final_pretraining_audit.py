"""训练前最终数据审计（只读，不修改任何原始数据）。

回应《最终数据冻结与正式建模准备》任务书第二/四/五节：
- 三个数据版本的主 Parquet 元数据与标签分布对比；
- v10.2（2026-10-25 交付）主表：主键唯一性、标签规则验证、抽样结构、黄金独立性；
- 16 个再分析降雨特征 + slot_2h + surge 的逐年 NaN（可得性审计输入）；
- diagnostic_full_days.parquet 全网格诊断集：标签口径、抽样内外分布差异（KS 检验）。

用法：.venv/bin/python scripts/final_pretraining_audit.py
输出：outputs/final_audit_20261025/*.json / *.csv（只新增，不覆盖历史产物）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = str(Path(__file__).resolve().parents[1])  # 仓库根目录（可移植，勿改回绝对路径）
DATA = f"{BASE}/data"
OUT = f"{BASE}/outputs/final_audit_20261025"
os.makedirs(OUT, exist_ok=True)

MAIN = {
    "v10.0": f"{DATA}/delivery_v10/flood_risk_dataset_2h_500m.parquet",
    "v10.1": f"{DATA}/delivery_v10_1/flood_risk_dataset_2h_500m.parquet",
    "v10.2": f"{DATA}/delivery_v10_20261025/flood_risk_dataset_2h_500m.parquet",
}
DIAG = f"{DATA}/delivery_v10_20261025/diagnostic_full_days.parquet"
GOLDEN_MAP = f"{DATA}/delivery_v10_20261025/golden_grid_map_500m.csv"


def meta(path: str) -> dict:
    pf = pq.ParquetFile(path)
    md = pf.metadata
    return {
        "rows": md.num_rows,
        "cols": md.num_columns,
        "row_groups": md.num_row_groups,
        "columns": pf.schema_arrow.names,
    }


def small_table(path: str, cols: list[str]) -> pd.DataFrame:
    cats = [c for c in ("grid_id", "date", "split", "sample_role", "label_source",
                        "attrib_conf", "confidence", "tide_station") if c in cols]
    return pq.read_table(path, columns=cols).to_pandas(categories=cats)


def year_of(date_cat: pd.Categorical) -> np.ndarray:
    years = date_cat.categories.astype(str).str[:4].to_numpy()
    return years[date_cat.codes.to_numpy()]


def label_dist_by_split(path: str) -> dict:
    cols = ["grid_id", "date", "slot", "flooded_2h", "split"]
    df = small_table(path, cols)
    dates = sorted(df["date"].cat.categories.astype(str))
    out = {"n_rows": int(len(df)),
           "date_min": dates[0], "date_max": dates[-1]}
    g = df.groupby("split", observed=True)["flooded_2h"]
    rows = g.agg(rows="size", pos="sum")
    out["by_split"] = {str(k): {"rows": int(r["rows"]), "pos": int(r["pos"]),
                                "pos_rate": float(r["pos"] / r["rows"])}
                       for k, r in rows.iterrows()}
    dup = df.duplicated(["grid_id", "date", "slot"]).sum()
    out["duplicate_keys"] = int(dup)
    del df
    return out


# ---------------------------------------------------------------------------
# 1. 版本元数据 + 标签分布
# ---------------------------------------------------------------------------
result: dict = {"versions": {}}
for ver, path in MAIN.items():
    m = meta(path)
    print(f"[{ver}] rows={m['rows']:,} cols={m['cols']}")
    m["label"] = label_dist_by_split(path)
    result["versions"][ver] = m

with open(f"{OUT}/version_meta.json", "w") as f:
    json.dump(result["versions"], f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 2. v10.2 深度审计（标签规则 / 抽样 / 黄金）
# ---------------------------------------------------------------------------
cols2 = ["grid_id", "date", "slot", "flooded", "flooded_2h", "flooded_2h_peak",
         "is_A", "is_event_day", "warn_level", "warn_level_day",
         "split", "sample_role", "label_source", "attrib_conf",
         "is_golden", "golden_overrides_rule", "dc", "cv_fold"]
df = small_table(MAIN["v10.2"], cols2)
print(f"v10.2 loaded: {len(df):,} rows")

deep: dict = {}
deep["n_rows"] = int(len(df))
deep["date_min"], deep["date_max"] = sorted(df["date"].cat.categories.astype(str))[0], \
    sorted(df["date"].cat.categories.astype(str))[-1]
deep["duplicate_keys"] = int(df.duplicated(["grid_id", "date", "slot"]).sum())
deep["n_grids"] = int(df["grid_id"].nunique())
deep["n_dates"] = int(df["date"].nunique())

# 2a. split × 样本角色
ct = pd.crosstab(df["split"], df["sample_role"])
deep["split_x_sample_role"] = {str(i): {str(c): int(v) for c, v in r.items()}
                               for i, r in ct.iterrows()}
ct.to_csv(f"{OUT}/split_x_sample_role.csv")

# 2b. 标签规则验证
isA = df["is_A"].astype("int8").to_numpy()
evd = df["is_event_day"].astype("int8").to_numpy()
y = df["flooded_2h"].astype("int8").to_numpy()
wl = df["warn_level"].astype("int8").to_numpy()
gold = df["is_golden"].astype("int8").to_numpy()
rule_pos = (isA == 1) & (evd == 1)
deep["rule_check"] = {
    "pos_total": int(y.sum()),
    "pos_and_rule": int((y == 1)[rule_pos].sum()),
    "pos_but_not_A": int(((y == 1) & (isA == 0)).sum()),
    "pos_but_nonevent": int(((y == 1) & (evd == 0)).sum()),
    "pos_but_notA_notgolden": int(((y == 1) & (isA == 0) & (gold == 0)).sum()),
    "pos_but_nonevent_notgolden": int(((y == 1) & (evd == 0) & (gold == 0)).sum()),
    "pos_warn_level_lt2": int(((y == 1) & (wl < 2)).sum()),
    "pos_warn_level_lt2_notgolden": int(((y == 1) & (wl < 2) & (gold == 0)).sum()),
    "pos_warn_level_eq0": int(((y == 1) & (wl == 0)).sum()),
    "rule_share_of_pos": float((y == 1)[rule_pos].sum() / max((y == 1).sum(), 1)),
    "isA0_rows_pos_rate": float(y[isA == 0].mean()),
    "isA1_rows_pos_rate": float(y[isA == 1].mean()),
    "nonevent_rows_pos": int(y[evd == 0].sum()),
    "warn_lt2_rows_pos_rate": float(y[wl < 2].mean()),
}
pd.crosstab(df["attrib_conf"], df["flooded_2h"]).to_csv(f"{OUT}/attrib_conf_x_label.csv")
deep["flooded_2h_peak"] = {"n_pos": int(df["flooded_2h_peak"].sum()),
                           "n_neg": int((df["flooded_2h_peak"] == 0).sum()),
                           "n_other": int((~df["flooded_2h_peak"].isin([0, 1])).sum())}

# 2c. 抽样结构
event_days = sorted(df.loc[df["is_event_day"] == 1, "date"].unique())
nonevent_days = sorted(df.loc[df["is_event_day"] == 0, "date"].unique())
per_day = df.groupby("date", observed=True).agg(
    rows=("flooded_2h", "size"), grids=("grid_id", "nunique"),
    a_share=("is_A", "mean"), pos=("flooded_2h", "sum"),
    event=("is_event_day", "max"))
ev = per_day[per_day["event"] == 1]
nev = per_day[per_day["event"] == 0]
deep["sampling"] = {
    "event_days": len(ev), "nonevent_days": len(nev),
    "event_day_rows_min_max": [int(ev["rows"].min()), int(ev["rows"].max())],
    "event_day_grids_mean": float(ev["grids"].mean()),
    "event_day_A_share_mean": float(ev["a_share"].mean()),
    "nonevent_rows_per_day_unique": sorted(nev["rows"].unique().tolist())[:5],
    "nonevent_grids_per_day_unique": sorted(nev["grids"].unique().tolist())[:5],
    "nonevent_A_share_min_mean_max": [float(nev["a_share"].min()),
                                      float(nev["a_share"].mean()),
                                      float(nev["a_share"].max())],
    "nonevent_A_share_std": float(nev["a_share"].std()),
    "nonevent_A_share_q10_q90": [float(nev["a_share"].quantile(0.1)),
                                 float(nev["a_share"].quantile(0.9))],
}
per_day.to_csv(f"{OUT}/per_day_sampling.csv")
# 全网格 A 类占比（来自静态表）用于对照
static = pd.read_csv(f"{DATA}/delivery_v10_20261025/grid_500m_static.csv",
                     usecols=["grid_id", "is_A"])
deep["sampling"]["fullgrid_A_share"] = float(static["is_A"].mean())
deep["sampling"]["fullgrid_grids"] = int(len(static))

# 2d. 黄金独立性
gm = pd.read_csv(GOLDEN_MAP)
golden_rows = df[df["split"] == "golden_eval"]
deep["golden"] = {
    "golden_map_rows": int(len(gm)),
    "golden_map_dates": sorted(gm["event_date"].astype(str).unique().tolist()),
    "golden_map_grids": int(gm["grid_id"].nunique()),
    "golden_eval_rows": int(len(golden_rows)),
    "golden_eval_dates": sorted(golden_rows["date"].astype(str).unique().tolist()),
    "golden_eval_grids": int(golden_rows["grid_id"].nunique()),
    "golden_eval_pos": int(golden_rows["flooded_2h"].sum()),
    "golden_eval_overrides": int(golden_rows["golden_overrides_rule"].sum()),
    "golden_eval_sample_role": golden_rows["sample_role"].value_counts().to_dict(),
    "golden_eval_pos_notwarn": int(((golden_rows["flooded_2h"] == 1) & (golden_rows["warn_level"] < 2)).sum()),
}
# 跨划分同 (grid,date) 重复？
gd_all = df[["grid_id", "date", "split"]].drop_duplicates()
gg = gd_all[gd_all["split"] == "golden_eval"][["grid_id", "date"]]
tr = gd_all[gd_all["split"] == "train"][["grid_id", "date"]]
te = gd_all[gd_all["split"] == "test"][["grid_id", "date"]]
deep["golden"]["grid_day_overlap_with_train"] = int(len(gg.merge(tr)))
deep["golden"]["grid_day_overlap_with_test"] = int(len(gg.merge(te)))
gset = set(golden_rows["grid_id"].astype(str).unique())
deep["golden"]["grids_also_in_train"] = int(
    len(gset & set(df.loc[df["split"] == "train", "grid_id"].astype(str).unique())))
deep["golden"]["grids_also_in_test"] = int(
    len(gset & set(df.loc[df["split"] == "test", "grid_id"].astype(str).unique())))
deep["golden"]["train_rows_on_2018_09_16"] = int((df["date"] == "2018-09-16").sum())
deep["golden"]["test_rows_on_2023_09_08"] = int((df["date"] == "2023-09-08").sum())
del df

# ---------------------------------------------------------------------------
# 3. 逐年 NaN（可得性审计输入）
# ---------------------------------------------------------------------------
rain_cols = ["era5_rain_prev3h_mm", "era5_rain_prev6h_mm", "era5_rain_prev12h_mm",
             "era5_rain_prev24h_mm", "era5_rain_prev48h_mm", "era5_rain_max1h_prev6h_mm",
             "era5_rain_max1h_prev24h_mm", "era5_api72_mm",
             "power_rain_prev3h_mm", "power_rain_prev6h_mm", "power_rain_prev12h_mm",
             "power_rain_prev24h_mm", "power_rain_prev48h_mm", "power_rain_max1h_prev6h_mm",
             "power_rain_max1h_prev24h_mm", "power_api72_mm",
             "era5_rain_slot_2h_mm", "power_rain_slot_2h_mm",
             "surge_max_m", "tide_pred_max_m", "tide_pred_mean_m"]
tbl = pq.read_table(MAIN["v10.2"], columns=["date"] + rain_cols)
years = pc.utf8_slice_codeunits(tbl.column("date"), 0, 4)
nan_by_year: dict = {}
for yr in sorted(pc.unique(years).to_pylist()):
    mask = pc.equal(years, yr)
    sub = tbl.filter(mask)
    nan_by_year[yr] = {c: sub.column(c).null_count / sub.num_rows for c in rain_cols}
nan_df = pd.DataFrame(nan_by_year).T
nan_df.index.name = "year"
nan_df.to_csv(f"{OUT}/nan_rate_by_year.csv")
deep["nan_rate_by_year"] = {yr: {c: round(float(v), 6) for c, v in row.items()}
                            for yr, row in nan_by_year.items()}
del tbl

with open(f"{OUT}/v10_2_deep_audit.json", "w") as f:
    json.dump(deep, f, ensure_ascii=False, indent=2, default=str)

# ---------------------------------------------------------------------------
# 4. 诊断集审计 + 抽样内外分布对比
# ---------------------------------------------------------------------------
d = pd.read_parquet(DIAG)
diag: dict = {"rows": int(len(d)), "cols": int(d.shape[1]),
              "dates": sorted(str(x) for x in d["date"].unique()),
              "grids": int(d["grid_id"].nunique())}
diag["label_cols_allzero"] = {
    c: bool((d[c] == 0).all()) for c in ("weak_label_flooded_2h", "label_verified")}
diag["weak_peak_values"] = sorted(d["weak_label_flooded_2h_peak"].unique().tolist())
diag["era5_nan_total"] = int(d[[c for c in d.columns if c.startswith("era5_")]].isna().sum().sum())
diag["in_main_share"] = {str(k): float(v) for k, v in
                         d.groupby("date", observed=True)["in_main_dataset"].mean().items()}

feat_cmp = ["elevation", "impervious_frac", "era5_rain_prev24h_mm",
            "power_rain_prev24h_mm", "tide_pred_max_m"]
rows = []
from scipy.stats import ks_2samp  # noqa: E402

for dt, g in d.groupby("date", observed=True):
    ins = g[g["in_main_dataset"] == 1]
    outs = g[g["in_main_dataset"] == 0]
    for feat in feat_cmp + ["is_A"]:
        a, b = ins[feat].dropna(), outs[feat].dropna()
        ks = ks_2samp(a, b) if len(a) > 5 and len(b) > 5 else None
        rows.append({
            "date": str(dt), "feature": feat,
            "mean_sampled_35": float(a.mean()), "mean_full_5482": float(b.mean()),
            "std_sampled": float(a.std()), "std_full": float(b.std()),
            "ks_stat": None if ks is None else float(ks.statistic),
            "ks_p": None if ks is None else float(ks.pvalue),
        })
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(f"{OUT}/diagnostic_sampling_bias.csv", index=False)
diag["sampling_bias_ks_significant"] = int((cmp_df["ks_p"] < 0.05).sum())
with open(f"{OUT}/diagnostic_audit.json", "w") as f:
    json.dump(diag, f, ensure_ascii=False, indent=2)

print(json.dumps({"deep_keys": list(deep.keys()), "diag": diag}, ensure_ascii=False, indent=2))
print("DONE -> outputs/final_audit_20261025/")
