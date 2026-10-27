"""v10.1（第二批交付）核查：与 v10.0 的差异 + 首轮四类问题是否解决。

只读核查，不改动任何已有模型与结果。输出 outputs/v10_1_check/summary.json。

用法：
    .venv/bin/python scripts/check_delivery_v10_1.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from floodrisk import realdata as rd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/delivery_v10/flood_risk_dataset_2h_500m.parquet"
B2 = ROOT / "data/delivery_v10_1"
NEW = B2 / "flood_risk_dataset_2h_500m.parquet"
OUT = ROOT / "outputs/v10_1_check"

KEYS = ["grid_id", "date", "slot"]
DYN = [
    "split", "is_event_day", "is_A", "warn_level", "flooded_2h",
    "tide_pred_max_m", "surge_max_m",
    "power_rain_slot_2h_mm", "power_rain_prev6h_mm", "power_rain_prev12h_mm",
    "power_rain_prev24h_mm", "power_rain_prev48h_mm",
    "era5_rain_slot_2h_mm", "era5_rain_prev6h_mm", "era5_rain_prev12h_mm",
    "era5_rain_prev24h_mm", "era5_rain_prev48h_mm",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def anchoring_mae(d: pd.DataFrame, col: str, hours: int) -> dict:
    """start = 前 n 槽（不含本槽）累加；end = 含本槽累加。n = hours//2。"""
    n = hours // 2
    src = col.split("_rain_")[0]  # power / era5
    g = d.groupby("grid_id", observed=True)[f"{src}_rain_slot_2h_mm"]
    start = g.transform(lambda s: s.shift(1).rolling(n, min_periods=n).sum())
    end = g.transform(lambda s: s.rolling(n, min_periods=n).sum())
    m = start.notna() & d[col].notna()
    mae_s = float((d.loc[m, col] - start[m]).abs().mean()) if m.any() else None
    m2 = end.notna() & d[col].notna()
    mae_e = float((d.loc[m2, col] - end[m2]).abs().mean()) if m2.any() else None
    return {"col": col, "hours": hours, "n": int(m.sum()), "mae_start": mae_s, "mae_end": mae_e}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep: dict = {"files": {}, "parquet": {}, "diff_vs_v10_0": {}, "anchoring": [], "stpu": {}, "static": {}, "golden_map": {}}

    # ---- 0. 文件级 ----
    rep["files"] = {
        "old_sha256": sha256(OLD),
        "new_sha256": sha256(NEW),
        "zip_sha256": "7d643362199935b195ec2da1a7021223e20400c3f09d32c05e8cd3d4adffef3f",
    }

    # ---- 1. schema / 行数 ----
    cols_old = pq.read_schema(str(OLD)).names
    cols_new = pq.read_schema(str(NEW)).names
    meta = pq.ParquetFile(str(NEW)).metadata
    rep["parquet"]["schema"] = {
        "n_cols_old": len(cols_old), "n_cols_new": len(cols_new),
        "same_names": cols_old == cols_new,
        "added": sorted(set(cols_new) - set(cols_old)),
        "removed": sorted(set(cols_old) - set(cols_new)),
        "rows": meta.num_rows,
    }

    # ---- 2. 关键列全表统计 ----
    new = rd.load_delivery(str(NEW), columns=KEYS + DYN, categorical=["grid_id", "date", "split"])
    new["year"] = new["date"].astype(object).astype(str).str[:4]
    rep["parquet"]["rows_loaded"] = len(new)
    rep["parquet"]["dup_keys"] = int(new.duplicated(KEYS).sum())
    rep["parquet"]["date_range"] = [str(min(new["date"].cat.categories)), str(max(new["date"].cat.categories))]
    rep["parquet"]["n_grids"] = int(new["grid_id"].nunique())

    # 标签决定性
    rule = (new["is_A"].astype(int) == 1) & (new["warn_level"].astype(int) >= 2)
    y = new["flooded_2h"].astype(int)
    rep["parquet"]["label_determinism"] = round(float((rule == (y == 1)).mean()), 6)
    rep["parquet"]["positives"] = int((y == 1).sum())

    # split 计数与日期范围
    sp = {}
    for name, sub in new.groupby(new["split"].astype(str), observed=True):
        dstr = sub["date"].astype(object).astype(str)
        sp[name] = {
            "rows": int(len(sub)),
            "date_min": dstr.min(),
            "date_max": dstr.max(),
            "pos_rate": round(float(sub["flooded_2h"].mean()), 4),
        }
    rep["parquet"]["split"] = sp

    # ERA5 / 其他动态列 NaN 率（按年）
    na = {}
    for c in ["era5_rain_prev6h_mm", "era5_rain_slot_2h_mm", "surge_max_m", "tide_pred_max_m"]:
        na[c] = {yr: round(float(v), 4) for yr, v in new.groupby("year")[c].apply(lambda s: s.isna().mean()).items()}
    rep["parquet"]["na_rate_by_year"] = na

    # 降雨空间差异：事件日抽 30 个 (date,slot)，看每槽内不同取值的个数
    ev = new[new["is_event_day"] == 1]
    samp = ev[["date", "slot"]].drop_duplicates().sample(30, random_state=0)
    pw_nu, e5_nu = [], []
    for _, r in samp.iterrows():
        sub = ev[(ev["date"] == r["date"]) & (ev["slot"] == r["slot"])]
        pw_nu.append(int(sub["power_rain_slot_2h_mm"].nunique()))
        e5_nu.append(int(sub["era5_rain_slot_2h_mm"].nunique()))
    rep["parquet"]["spatial_variation_30slots"] = {
        "power_nunique_median": float(np.median(pw_nu)),
        "era5_nunique_median": float(np.median(e5_nu)),
    }

    # 非事件日网格覆盖
    ne = new[new["is_event_day"] == 0]
    per_day = ne.groupby(ne["date"].astype(str))["grid_id"].nunique()
    rep["parquet"]["nonevent_grids_per_day"] = {"median": float(per_day.median()), "max": int(per_day.max())}
    ev_day = ev.groupby(ev["date"].astype(str))["grid_id"].nunique()
    rep["parquet"]["event_grids_per_day"] = {"median": float(ev_day.median()), "max": int(ev_day.max())}
    rep["parquet"]["any_day_full_coverage"] = bool((ev_day.max() >= 5517) or (per_day.max() >= 5517))

    # ---- 3. 锚定复测（v10.1）----
    ev_days = sorted(ev["date"].astype(object).astype(str).unique())
    # 找最长连续事件日段
    best, cur = [ev_days[0]], [ev_days[0]]
    for d in ev_days[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(cur[-1])).days == 1:
            cur.append(d)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [d]
    if len(cur) > len(best):
        best = cur
    win = new[new["date"].astype(object).astype(str).isin(best)]
    cnt = win.groupby("grid_id", observed=True)["slot"].count()
    good = cnt[cnt == 12 * len(best)].index[:60]
    probe = win[win["grid_id"].isin(good)].sort_values(["grid_id", "timestamp"])
    rep["anchoring_window"] = {"dates": [best[0], best[-1]], "n_grids": len(good), "rows": len(probe)}
    for src in ("power", "era5"):
        for h in (6, 12, 24, 48):
            rep["anchoring"].append(anchoring_mae(probe, f"{src}_rain_prev{h}h_mm", h))

    # ---- 4. 与 v10.0 的逐值对比（锚定窗口内，非 ERA5 列是否变化）----
    keys = probe[KEYS].copy()
    cmp_cols = KEYS + ["is_A", "warn_level", "flooded_2h", "tide_pred_max_m",
                       "power_rain_slot_2h_mm", "power_rain_prev6h_mm", "era5_rain_prev6h_mm"]
    old_sub = rd.load_delivery(str(OLD), columns=cmp_cols, filter_keys=keys)
    new_sub = rd.load_delivery(str(NEW), columns=cmp_cols, filter_keys=keys)
    o = old_sub.sort_values(KEYS).reset_index(drop=True)
    n_ = new_sub.sort_values(KEYS).reset_index(drop=True)
    rep["diff_vs_v10_0"]["rows_compared"] = int(len(o))
    for c in cmp_cols:
        if c in KEYS:
            continue
        oc, nc = o[c].to_numpy(dtype="float64", na_value=np.nan), n_[c].to_numpy(dtype="float64", na_value=np.nan)
        both_nan = np.isnan(oc) & np.isnan(nc)
        same = (oc == nc) | both_nan
        rep["diff_vs_v10_0"][c] = {
            "same_rate": round(float(same.mean()), 6),
            "old_nan": int(np.isnan(oc).sum()), "new_nan": int(np.isnan(nc).sum()),
        }

    # ---- 5. STPU 映射 / 静态主表 / 黄金映射 ----
    to_stpu = pd.read_csv(B2 / "grid_500m_to_stpu.csv")
    overlap = pd.read_csv(B2 / "grid_500m_stpu_overlap.csv")
    static = pd.read_csv(B2 / "grid_500m_static.csv")
    gmap = pd.read_csv(B2 / "golden_grid_map_500m.csv")
    grids_main = set(new["grid_id"].cat.categories if isinstance(new["grid_id"].dtype, pd.CategoricalDtype) else new["grid_id"].unique())

    stpu_col = [c for c in to_stpu.columns if "stpu" in c.lower() or "STPU" in c]
    rep["stpu"] = {
        "to_stpu_rows": len(to_stpu), "to_stpu_cols": list(to_stpu.columns),
        "overlap_rows": len(overlap), "overlap_cols": list(overlap.columns),
        "grids_covered": int(to_stpu["grid_id"].isin(grids_main).sum()),
        "n_stpu": int(to_stpu[stpu_col[0]].nunique()) if stpu_col else None,
    }
    if "frac" in overlap.columns:
        fs = overlap.groupby("grid_id")["frac"].sum()
        rep["stpu"]["frac_sum_per_grid"] = {"min": round(float(fs.min()), 4), "max": round(float(fs.max()), 4)}

    rep["static"] = {
        "rows": len(static), "cols": len(static.columns),
        "grid_cover": int(static["grid_id"].isin(grids_main).sum()) if "grid_id" in static.columns else None,
        "has_all_main_features": sorted(set(rd.FEATURES) - set(static.columns) - {
            # 动态列不期望在静态表
            "power_rain_prev3h_mm", "power_rain_prev6h_mm", "power_rain_prev12h_mm",
            "power_rain_prev24h_mm", "power_rain_prev48h_mm", "power_rain_max1h_prev6h_mm",
            "power_rain_max1h_prev24h_mm", "power_api72_mm",
            "tide_pred_max_m", "tide_pred_mean_m", "tide_n_hours",
            "month", "hour_sin", "hour_cos",
        }),
    }
    rep["golden_map"] = {"rows": len(gmap), "cols": list(gmap.columns)}

    out = OUT / "summary.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str)[:6000])
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
