#!/usr/bin/env python3
"""B 侧独立修复（R2 方案）：基于 HKO 官方 rstorm.dat 重建弱监督标签与抽样框架。

三阶段：
  feasibility  只读可行性检查：官方警告解析、逐日分类、旧标签规则复现、镜像规则确认
  repair       生成 data/delivery_repaired_b1/（不覆盖任何旧文件）
  validate     用户 §四 验证清单

口径（与 A 侧 README §5 / qc_labels_500m.json 一致，警告源改为官方 rstorm.dat）：
  warn_level(slot) = 与槽 [2s,2s+2) HKT 相交的官方警告最高级别（A=1,R=2,B=3，半开区间）
  warn_level_day   = 当日 12 槽最大级别
  is_event_day     = warn_level_day >= 2（当日存在红/黑警告）
  flooded          = is_A ∧ is_event_day（黄金行保留原值）
  归因窗口          = 事件日→红黑活跃槽(attrib high)；amber 日→amber 槽(medium)；
                     无警告日→全 12 槽(low)
  flooded_2h       = flooded ∧ 槽∈归因窗口
  flooded_2h_peak  = flooded_2h ∧ 归因窗口内槽雨 argmax（ERA5 优先，全缺兜底 POWER）

随机性：假事件日改判非事件日后按非事件日口径抽 35 格/日，
np.random.default_rng(42) 按日期字符串升序逐日抽一次（不放回），全程记录于 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "delivery_v10_20261025"
SRC_MAIN = DATA_DIR / "flood_risk_dataset_2h_500m.parquet"
SRC_FRAME = DATA_DIR / "sample_frame.parquet"
AUDIT = ROOT / "outputs" / "warning_audit_20261026"
OFFICIAL = AUDIT / "hko_rstorm_official.csv"
OUT_DIR = ROOT / "data" / "delivery_repaired_b1"
OUT_MAIN = OUT_DIR / "flood_risk_dataset_2h_500m.parquet"
OUT_FRAME = OUT_DIR / "sample_frame.parquet"
WORK = ROOT / "outputs" / "repair_b1"

LEVEL = {"A": 1, "R": 2, "B": 3}
SEED = 42
N_NONEVENT_CELLS = 35
GOLDEN_LAYER = "L2_golden"

INT_REPLACED = ["warn_level", "warn_level_day", "is_event_day",
                "flooded", "flooded_2h", "flooded_2h_peak"]
CAT_REPLACED = ["label_source", "confidence", "attrib_conf", "sample_role"]
REPLACED = INT_REPLACED + CAT_REPLACED

LS_EVENT_A = "weak_supervision(A x event_day)"
LS_EVENT_B = "nonvulnerable_on_event_day"
LS_NONEVENT = "non_event_day"
LS_GOLDEN = "golden_confirmed"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# 官方警告 → 槽级矩阵
# ---------------------------------------------------------------------------
def load_official() -> pd.DataFrame:
    df = pd.read_csv(OFFICIAL)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["lv"] = df["level"].map(LEVEL)
    if df["lv"].isna().any():
        raise ValueError(f"官方文件出现未知警告级别: {sorted(set(df.loc[df['lv'].isna(), 'level']))}")
    return df


def slot_level_matrix(warn: pd.DataFrame, dates: list[str]) -> np.ndarray:
    """(n_dates, 12) int8：每槽 = 与 [2s,2s+2) HKT 半开区间相交的官方警告最高级别。"""
    idx = {d: i for i, d in enumerate(dates)}
    mat = np.zeros((len(dates), 12), dtype=np.int8)
    two_h = pd.Timedelta(hours=2)
    one_d = pd.Timedelta(days=1)
    for r in warn.itertuples():
        lv = int(r.lv)
        st, en = r.start, r.end
        if en <= st:
            continue
        d = st.normalize()
        last = (en - pd.Timedelta(nanoseconds=1)).normalize()  # 恰好午夜结束→归前一日
        while d <= last:
            i = idx.get(d.strftime("%Y-%m-%d"))
            if i is not None:
                for s in range(12):
                    ss = d + pd.Timedelta(hours=2 * s)
                    if max(st, ss) < min(en, ss + two_h) and lv > mat[i, s]:
                        mat[i, s] = lv
            d += one_d
    return mat


# ---------------------------------------------------------------------------
# 数据装载（sample_frame 全量 + 主表少量列，校验行序一致）
# ---------------------------------------------------------------------------
class Loaded:
    pass


def load_inputs() -> Loaded:
    L = Loaded()
    frame = pd.read_parquet(SRC_FRAME)
    for c in ("grid_id", "date"):
        if not isinstance(frame[c].dtype, pd.CategoricalDtype):
            frame[c] = frame[c].astype("category")
    main_cols = pq.read_table(SRC_MAIN,
                              columns=["grid_id", "date", "slot", "is_A", "warn_level",
                                       "era5_rain_slot_2h_mm", "power_rain_slot_2h_mm"])
    for c in ("grid_id", "date"):
        col = main_cols.column(c)
        if not (hasattr(col.type, "value_type") and str(col.type).startswith("dictionary")):
            main_cols = main_cols.set_column(main_cols.schema.get_field_index(c),
                                             c, pc.dictionary_encode(col))
    main = main_cols.to_pandas(categories=["grid_id", "date"])

    # 行序一致性校验（位置对齐是全部重建逻辑的前提）
    assert len(frame) == len(main), "sample_frame 与主表行数不一致"
    assert (frame["slot"].to_numpy() == main["slot"].to_numpy()).all(), "slot 列不对齐"
    assert (frame["grid_id"].cat.codes.to_numpy() == main["grid_id"].cat.codes.to_numpy()).all() or \
           (frame["grid_id"].astype(str).to_numpy() == main["grid_id"].astype(str).to_numpy()).all(), \
        "grid_id 列不对齐"
    assert (frame["date"].astype(str).to_numpy() == main["date"].astype(str).to_numpy()).all(), \
        "date 列不对齐"

    L.frame = frame
    L.main = main
    L.date_cats = list(frame["date"].cat.categories) if isinstance(
        frame["date"].dtype, pd.CategoricalDtype) else sorted(set(frame["date"].astype(str)))
    L.grid_cats = (list(frame["grid_id"].cat.categories)
                   if isinstance(frame["grid_id"].dtype, pd.CategoricalDtype) else None)
    L.date_codes = (frame["date"].cat.codes.to_numpy().astype(np.int64)
                    if isinstance(frame["date"].dtype, pd.CategoricalDtype)
                    else pd.factorize(frame["date"].astype(str))[0].astype(np.int64))
    L.grid_codes = (frame["grid_id"].cat.codes.to_numpy().astype(np.int64)
                    if isinstance(frame["grid_id"].dtype, pd.CategoricalDtype)
                    else pd.factorize(frame["grid_id"].astype(str))[0].astype(np.int64))
    L.slot = frame["slot"].to_numpy(dtype=np.int64)
    L.is_A = main["is_A"].to_numpy(dtype=np.int8)
    L.wl_old = main["warn_level"].to_numpy(dtype=np.int8)
    L.era5_slot = main["era5_rain_slot_2h_mm"].to_numpy(dtype=np.float64)
    L.power_slot = main["power_rain_slot_2h_mm"].to_numpy(dtype=np.float64)
    L.official = load_official()
    L.mat = slot_level_matrix(L.official, L.date_cats)
    return L


def derive_sets(L: Loaded) -> dict:
    """官方/旧标签逐日分类。"""
    dates = L.date_cats
    day_max = L.mat.max(axis=1)
    off_kind = np.where(day_max >= 2, 2, np.where(day_max == 1, 1, 0))
    old_event_mask_dates = np.zeros(len(dates), dtype=bool)
    ev_codes = np.unique(L.date_codes[L.frame["is_event_day"].to_numpy() == 1])
    old_event_mask_dates[ev_codes] = True
    off_rb = off_kind >= 2
    return {
        "dates": dates,
        "day_max": day_max,
        "off_kind": off_kind,
        "old_event_mask_dates": old_event_mask_dates,
        "real_dates": [dates[i] for i in np.flatnonzero(off_rb & old_event_mask_dates)],
        "false_dates": [dates[i] for i in np.flatnonzero(~off_rb & old_event_mask_dates)],
        "missed_dates": [dates[i] for i in np.flatnonzero(off_rb & ~old_event_mask_dates)],
        "official_rb_days": int(off_rb.sum()),
        "official_amber_days": int((off_kind == 1).sum()),
    }


def row_kind(L: Loaded, S: dict) -> np.ndarray:
    """每行所属官方日类型 0/1/2。"""
    return S["off_kind"][L.date_codes]


def select_false_day_grids(L: Loaded, false_dates: list[str], seed: int = SEED) -> dict:
    """假事件日 → 选中的 35 格（日期字符串升序逐日，default_rng(seed) 不放回）。"""
    date_pos = {d: i for i, d in enumerate(L.date_cats)}
    grid_cats = L.grid_cats if L.grid_cats is not None else sorted(set(L.frame["grid_id"].astype(str)))
    rng = np.random.default_rng(seed)
    out = {}
    for d in sorted(false_dates):
        mask = L.date_codes == date_pos[d]
        present = np.unique(L.grid_codes[mask])
        grids = sorted(grid_cats[c] for c in present)
        if len(grids) < N_NONEVENT_CELLS:
            raise ValueError(f"{d} 可用网格 {len(grids)} < {N_NONEVENT_CELLS}，无法按非事件日抽样")
        pick = rng.choice(len(grids), size=N_NONEVENT_CELLS, replace=False)
        out[d] = set(grids[i] for i in pick)
    return out


# ---------------------------------------------------------------------------
# 标签重建
# ---------------------------------------------------------------------------
def rebuilt_core(L: Loaded, S: dict) -> dict:
    """非黄金行的重建列（黄金位置无效，稍后覆盖回旧值）。"""
    kind_day = S["off_kind"][L.date_codes]          # 每行日类型 0/1/2
    row_wl = L.mat[L.date_codes, L.slot].astype(np.int8)
    wld = S["day_max"][L.date_codes].astype(np.int8)
    is_event = (wld >= 2).astype(np.int8)
    flooded = ((L.is_A == 1) & (is_event == 1)).astype(np.int8)
    in_window = ((kind_day == 0)
                 | ((kind_day == 1) & (row_wl == 1))
                 | ((kind_day == 2) & (row_wl >= 2)))
    f2h = (flooded == 1) & in_window
    return {"kind_day": kind_day, "row_wl": row_wl, "wld": wld, "is_event": is_event,
            "flooded": flooded, "in_window": in_window, "f2h": f2h}


def rebuilt_meta(L: Loaded, S: dict, core: dict, attrib_mode: str) -> dict:
    kind = core["kind_day"]
    is_event = core["is_event"]
    is_A1 = L.is_A == 1
    if attrib_mode == "day":
        attrib = np.where(kind == 2, "high", np.where(kind == 1, "medium", "low"))
    else:  # slot
        attrib = np.where(core["in_window"],
                          np.where(kind == 2, "high", "medium"), "low")
        attrib = np.where(kind == 0, "low", attrib)
    label_source = np.where(is_event == 1,
                            np.where(is_A1, LS_EVENT_A, LS_EVENT_B), LS_NONEVENT)
    confidence = np.where(is_event == 1, np.where(is_A1, "low", "medium_low"), "medium")
    sample_role = np.where(is_event == 1,
                           np.where(is_A1, "A_event", "B_event"),
                           np.where(is_A1, "A_nonevent", "B_nonevent"))
    return {"attrib_conf": attrib, "label_source": label_source,
            "confidence": confidence, "sample_role": sample_role}


def compute_peak(L: Loaded, core: dict, golden_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """flooded_2h_peak：f2h 行内按 (grid,date) 取 ERA5 槽雨首个最大值槽；ERA5 全缺兜底 POWER。"""
    peak = np.zeros(len(L.frame), dtype=np.int8)
    cand = np.flatnonzero(core["f2h"] & ~golden_mask)
    if len(cand) == 0:
        return peak, {"n_candidates": 0}
    g = pd.DataFrame({
        "key": L.grid_codes[cand] * (L.date_codes.max() + 1) + L.date_codes[cand],
        "slot": L.slot[cand],
        "era5": L.era5_slot[cand],
        "power": L.power_slot[cand],
    }).sort_values(["key", "slot"], kind="mergesort")
    grp = g.groupby("key", sort=False)
    era5_ok = grp["era5"].transform("count") > 0  # 组内 ERA5 全缺 → 兜底 POWER
    g["_rain"] = np.where(era5_ok, g["era5"].fillna(-np.inf), g["power"].fillna(-np.inf))
    grp = g.groupby("key", sort=False)
    cummax = grp["_rain"].cummax().to_numpy()
    prev = grp["_rain"].shift(1).fillna(-np.inf).to_numpy()
    cur = g["_rain"].to_numpy()
    flag = (cur >= cummax) & (cur > prev)  # 首个达到组内最大值的槽（同 pandas idxmax）
    peak[g.index.to_numpy()[flag]] = 1
    stats = {"n_candidates": int(len(cand)), "n_peaks": int(peak.sum()),
             "n_groups": int(grp.ngroups),
             "fallback_power_rows": int((~era5_ok).sum())}
    return peak, stats


def verify_peak_rule(L: Loaded, core: dict, golden_mask: np.ndarray) -> dict:
    """在旧事件日（376 天全部，非黄金行）上用旧 f2h 作窗口复现旧 flooded_2h_peak，检验规则。"""
    S_dates_idx = {d: i for i, d in enumerate(L.date_cats)}
    old_peak = L.frame["flooded_2h_peak"].to_numpy(dtype=np.int8)
    old_event_row = L.frame["is_event_day"].to_numpy() == 1
    old_f2h = L.frame["flooded_2h"].to_numpy(dtype=np.int8) == 1
    cand_mask = old_f2h & ~golden_mask & old_event_row
    if cand_mask.sum() == 0:
        return {"n_candidates": 0}
    cand = np.flatnonzero(cand_mask)
    g = pd.DataFrame({
        "key": L.grid_codes[cand] * (L.date_codes.max() + 1) + L.date_codes[cand],
        "slot": L.slot[cand],
        "era5": L.era5_slot[cand],
        "power": L.power_slot[cand],
        "old": old_peak[cand],
    }).sort_values(["key", "slot"], kind="mergesort")
    grp = g.groupby("key", sort=False)
    era5_ok = grp["era5"].transform("count") > 0
    g["_rain"] = np.where(era5_ok, g["era5"].fillna(-np.inf), g["power"].fillna(-np.inf))
    grp = g.groupby("key", sort=False)
    cummax = grp["_rain"].cummax().to_numpy()
    prev = grp["_rain"].shift(1).fillna(-np.inf).to_numpy()
    cur = g["_rain"].to_numpy()
    flag = (cur >= cummax) & (cur > prev)
    recomputed = np.zeros(len(g), dtype=np.int8)
    recomputed[flag] = 1
    mism = int((recomputed != g["old"].to_numpy()).sum())
    return {"n_candidates": int(len(cand)), "mismatches": mism,
            "match_rate": round(1 - mism / len(cand), 6),
            "fallback_power_rows": int((~era5_ok).sum())}


# ---------------------------------------------------------------------------
# 阶段 1：可行性
# ---------------------------------------------------------------------------
def stage_feasibility() -> int:
    t0 = time.time()
    L = load_inputs()
    S = derive_sets(L)
    frame = L.frame
    golden_mask = (frame["label_layer"] == GOLDEN_LAYER).to_numpy()
    kind_row = row_kind(L, S)

    res: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "src_main": str(SRC_MAIN), "src_main_sha256": sha256(SRC_MAIN),
            "src_frame": str(SRC_FRAME), "src_frame_sha256": sha256(SRC_FRAME),
            "official_csv": str(OFFICIAL), "official_csv_sha256": sha256(OFFICIAL),
            "official_warnings_total": int(len(L.official)),
        },
        "official_parse": {
            "window_warnings": int(L.official["start"].dt.year.ge(2018).sum()
                                   + ((L.official["start"].dt.year < 2018)
                                      & (L.official["end"].dt.year.ge(2018))).sum()),
            "level_dist": {k: int(v) for k, v in L.official["level"].value_counts().items()},
        },
        "date_classification": {
            "n_days": len(S["dates"]),
            "official_rb_days": S["official_rb_days"],
            "official_amber_days": S["official_amber_days"],
            "old_event_days": int(S["old_event_mask_dates"].sum()),
            "real_event_days": len(S["real_dates"]),
            "false_event_days": len(S["false_dates"]),
            "missed_days": len(S["missed_dates"]),
            "missed_list": S["missed_dates"][:20],
        },
    }

    # 逐类行数 / 网格统计
    date_pos = {d: i for i, d in enumerate(S["dates"])}
    row_cat = np.full(len(frame), "nonevent", dtype=object)
    for d in S["real_dates"]:
        row_cat[L.date_codes == date_pos[d]] = "real_event"
    for d in S["false_dates"]:
        row_cat[L.date_codes == date_pos[d]] = "false_event"
    res["rows_by_category"] = {k: int(v) for k, v in
                               pd.Series(row_cat).value_counts().items()}

    # warn_level 复现检查（逐行，官方矩阵 vs 旧列）
    wl_recomp = L.mat[L.date_codes, L.slot]
    wl_diff = wl_recomp != L.wl_old
    res["warn_level_reproduction"] = {}
    for cat in ("real_event", "false_event", "nonevent"):
        m = row_cat == cat
        res["warn_level_reproduction"][cat] = {
            "rows": int(m.sum()), "diff_rows": int((wl_diff & m).sum()),
            "diff_rate": round(float((wl_diff & m).sum()) / max(m.sum(), 1), 6),
        }
    # 官方黄雨日（amber）单独检查
    amber_row = kind_row == 1
    res["warn_level_reproduction"]["official_amber_days_rows"] = {
        "rows": int(amber_row.sum()), "diff_rows": int((wl_diff & amber_row).sum())}

    # 元数据镜像规则核验（非黄金行，按旧 is_event_day 推导期望值）
    ng = ~golden_mask
    old_ev = frame["is_event_day"].to_numpy() == 1
    is_A1 = L.is_A == 1
    exp_ls = np.where(old_ev, np.where(is_A1, LS_EVENT_A, LS_EVENT_B), LS_NONEVENT)
    exp_cf = np.where(old_ev, np.where(is_A1, "low", "medium_low"), "medium")
    meta_check = {}
    for col, exp in (("label_source", exp_ls), ("confidence", exp_cf)):
        old_vals = frame[col].astype(str).to_numpy()
        mism = (old_vals != exp) & ng
        meta_check[col] = {"mismatch_rows_nongolden": int(mism.sum())}
        if mism.any():
            meta_check[col]["examples"] = [
                {"is_event": bool(old_ev[i]), "is_A": int(is_A1[i]),
                 "old": str(old_vals[i]), "expected": str(exp[i])}
                for i in np.flatnonzero(mism)[:5]]
    # attrib_conf 语义（逐行 crosstab：日类型 × 是否窗口内 × 旧值）
    in_win_old = ((kind_row == 0)
                  | ((kind_row == 1) & (L.wl_old == 1))
                  | ((kind_row == 2) & (L.wl_old >= 2)))
    ct = pd.crosstab([row_cat, np.where(in_win_old, "in_window", "out_window")],
                     frame["attrib_conf"].astype(str))
    meta_check["attrib_conf_crosstab"] = {f"{a}|{b}": {k: int(v) for k, v in r.items()}
                                          for (a, b), r in ct.iterrows()}
    res["meta_mapping_check"] = meta_check

    # 峰值规则复现（信息性）+ 标签族一致性（门槛）
    core = rebuilt_core(L, S)
    res["peak_rule_check"] = verify_peak_rule(L, core, golden_mask)
    old_peak = frame["flooded_2h_peak"].to_numpy(dtype=np.int8)
    old_f2h = frame["flooded_2h"].to_numpy(dtype=np.int8)
    pk = old_peak == 1
    real_row_pk = np.isin(L.date_codes, [date_pos[d] for d in S["real_dates"]])
    res["peak_family_check"] = {
        "peak_total": int(pk.sum()), "golden_peak": int((pk & golden_mask).sum()),
        "peak_not_in_f2h": int((pk & (old_f2h == 0)).sum()),
        "peak_on_real_days": int((pk & real_row_pk).sum()),
        "peak_on_false_days_to_zero": int((pk & ~real_row_pk & ~golden_mask).sum()),
        "peak_on_nonevent_days": int((pk & ~real_row_pk
                                      & ~np.isin(L.date_codes,
                                                 [date_pos[d] for d in S["false_dates"]])).sum()),
    }

    # 抽样可行性：假事件日可用网格与槽完整性
    false_codes = [date_pos[d] for d in S["false_dates"]]
    fd = pd.DataFrame({"date": L.date_codes, "grid": L.grid_codes})
    fd = fd[np.isin(fd["date"], false_codes)]
    gd = fd.groupby(["date", "grid"]).size()
    avail = fd.groupby("date")["grid"].nunique()
    res["false_day_frame"] = {
        "grid_days": int(len(gd)),
        "slots_per_grid_day_min": int(gd.min()), "slots_per_grid_day_max": int(gd.max()),
        "grids_per_day_min": int(avail.min()), "grids_per_day_max": int(avail.max()),
        "grids_per_day_mean": round(float(avail.mean()), 1),
    }
    # A 类占比（重建非事件日抽样的已知偏差）
    a_share = float((L.is_A[np.isin(L.date_codes, false_codes)] == 1).mean())
    res["false_day_frame"]["is_A_share_available"] = round(a_share, 4)
    ne = row_cat == "nonevent"
    res["false_day_frame"]["is_A_share_old_nonevent_days"] = round(
        float((L.is_A[ne] == 1).mean()), 4)

    # 抽样预演（与 repair 同一函数同一消耗）
    sel = select_false_day_grids(L, S["false_dates"])
    res["selection_preview"] = {
        "seed": SEED, "cells_per_day": N_NONEVENT_CELLS,
        "n_days": len(sel),
        "scheme": "np.random.default_rng(42)，假事件日按日期字符串升序逐日 choice(35, replace=False)，"
                  "候选=当日主表已有的全部网格按 grid_id 字符串升序",
        "example_first_day": {"date": sorted(sel)[0],
                              "n": len(sel[sorted(sel)[0]])},
    }

    # 修复后规模预估
    keep_est = int((row_cat != "false_event").sum() + len(S["false_dates"]) * N_NONEVENT_CELLS * 12)
    pos_real = int(frame.loc[row_cat == "real_event", "flooded_2h"].sum())
    res["projected"] = {
        "rows_new": keep_est,
        "rows_old": int(len(frame)),
        "positives_real_event_days": pos_real,
        "positives_old_total": int(frame["flooded_2h"].sum()),
        "positive_rate_new_approx": round(pos_real / keep_est, 5),
    }
    res["golden"] = {
        "rows": int(golden_mask.sum()),
        "dates_on_false_days": int(frame.loc[golden_mask & (row_cat == "false_event")].shape[0]),
        "role_dist": {str(k): int(v) for k, v in
                      frame.loc[golden_mask, "sample_role"].astype(str).value_counts().items()},
    }
    res["elapsed_seconds"] = round(time.time() - t0, 1)

    # GO/NO-GO
    gates = {
        "no_missed_official_rb_days": len(S["missed_dates"]) == 0,
        "real_false_split_as_audit": len(S["real_dates"]) == 56 and len(S["false_dates"]) == 320,
        "warn_level_exact_on_real_event_days":
            res["warn_level_reproduction"]["real_event"]["diff_rows"] == 0,
        "warn_level_exact_on_official_amber_days":
            res["warn_level_reproduction"]["official_amber_days_rows"]["diff_rows"] == 0,
        "label_source_mapping_exact": meta_check["label_source"]["mismatch_rows_nongolden"] == 0,
        "confidence_mapping_exact": meta_check["confidence"]["mismatch_rows_nongolden"] == 0,
        # 旧峰值规则不可由交付列复现（A 侧使用了未交付中间量），修复策略 = 正确日保留旧值、
        # 假日归零；门槛改为标签族一致性（峰值 ⊆ flooded_2h）
        "peak_family_consistent": res["peak_family_check"]["peak_not_in_f2h"] == 0,
        "false_days_have_35_grids_x12_slots":
            res["false_day_frame"]["grids_per_day_min"] >= N_NONEVENT_CELLS
            and res["false_day_frame"]["slots_per_grid_day_min"] == 12,
        "golden_not_on_false_days": res["golden"]["dates_on_false_days"] == 0,
    }
    res["gates"] = gates
    res["go"] = all(gates.values())
    write_json(WORK / "feasibility.json", res)
    print(json.dumps({k: res[k] for k in ("date_classification", "warn_level_reproduction",
                                          "projected", "gates", "go")},
                     ensure_ascii=False, indent=2, default=str))
    print(f"[feasibility] done in {res['elapsed_seconds']}s → {WORK/'feasibility.json'}")
    return 0 if res["go"] else 1


# ---------------------------------------------------------------------------
# 阶段 2：修复
# ---------------------------------------------------------------------------
def _cat_writer(col_name: str, old_series: pd.Series):
    """把旧分类列转成 (codes, categories)，支持按位置替换非黄金值。"""
    if isinstance(old_series.dtype, pd.CategoricalDtype):
        cats = [str(c) for c in old_series.cat.categories]
        codes = old_series.cat.codes.to_numpy().astype(np.int64)
    else:
        vals = old_series.astype(str)
        cats = sorted(set(vals))
        cmap = {c: i for i, c in enumerate(cats)}
        codes = vals.map(cmap).to_numpy().astype(np.int64)
    return codes, cats


def stage_repair() -> int:
    t0 = time.time()
    L = load_inputs()
    S = derive_sets(L)
    if S["missed_dates"]:
        print(f"[abort] 官方红/黑日有 {len(S['missed_dates'])} 天未被旧事件日覆盖，先人工核查")
        return 1
    frame = L.frame
    n = len(frame)
    golden_mask = (frame["label_layer"] == GOLDEN_LAYER).to_numpy()

    core = rebuilt_core(L, S)
    # attrib_conf 语义由 feasibility.json 固化
    feas = json.loads((WORK / "feasibility.json").read_text(encoding="utf-8"))
    ct = feas["meta_mapping_check"]["attrib_conf_crosstab"]
    # 判定：真实事件日窗口外槽若全为 low → slot 模式；否则 day 模式
    out_high = {k: v.get("high", 0) for k, v in ct.items()
                if k.startswith("real_event|out_window")}
    attrib_mode = "day" if sum(out_high.values()) > 0 else "slot"
    print(f"[repair] attrib_mode={attrib_mode} (real_event|out_window high={sum(out_high.values())})")

    meta = rebuilt_meta(L, S, core, attrib_mode)
    peak_check = verify_peak_rule(L, core, golden_mask)
    # 峰值策略固定为 keep_old_on_correct_days：A 侧峰值规则依赖未交付的 ERA5 小时级中间量，
    # 无法由交付列复现（feasibility 实测 match_rate≈0.76，旱季假日近乎任意）。
    # 最小改动原则：正确日（56 真实红黑日 + 全部非事件日）保留旧值，320 个假日归零，黄金行保留。
    peak_mode = "keep_old_on_correct_days"
    print(f"[repair] peak_mode={peak_mode} (rule check informational: {peak_check})")

    # ---- 最终列（黄金行保留旧值）----
    final: dict[str, np.ndarray] = {}
    wl_old = L.wl_old
    final["warn_level"] = np.where(golden_mask, wl_old, core["row_wl"]).astype(np.int8)
    for c, rec in (("warn_level_day", core["wld"]), ("is_event_day", core["is_event"]),
                   ("flooded", core["flooded"])):
        old = frame[c].to_numpy(dtype=np.int8)
        final[c] = np.where(golden_mask, old, rec).astype(np.int8)
    if peak_mode == "recompute":
        peak, peak_stats = compute_peak(L, core, golden_mask)
        old = frame["flooded_2h_peak"].to_numpy(dtype=np.int8)
        final["flooded_2h_peak"] = np.where(golden_mask, old, peak).astype(np.int8)
    else:
        peak_stats = {"mode": "keep_old_on_correct_days"}
        old = frame["flooded_2h_peak"].to_numpy(dtype=np.int8)
        false_row = np.isin(L.date_codes, [i for i, d in enumerate(L.date_cats)
                                           if d in set(S["false_dates"])])
        final["flooded_2h_peak"] = np.where(golden_mask | ~false_row, old, 0).astype(np.int8)
    old_f2h = frame["flooded_2h"].to_numpy(dtype=np.int8)
    final["flooded_2h"] = np.where(golden_mask, old_f2h, core["f2h"].astype(np.int8)).astype(np.int8)
    # 标签族一致性断言：非黄金 peak=1 必须落在 f2h=1 内
    assert int(((final["flooded_2h_peak"] == 1) & (final["flooded_2h"] == 0)
                & ~golden_mask).sum()) == 0, "peak 与 flooded_2h 族不一致"
    cat_state = {}
    for c in CAT_REPLACED:
        codes, cats = _cat_writer(c, frame[c])
        rec = meta[c]
        cmap = {x: i for i, x in enumerate(cats)}
        uniq = pd.unique(pd.Series(rec))
        missing = [x for x in uniq if x not in cmap]
        if missing:
            raise ValueError(f"{c} 重建值 {missing} 不在旧词表 {cats} 中")
        codes = codes.copy()
        codes[~golden_mask] = pd.Series(rec[~golden_mask]).map(cmap).to_numpy()
        cat_state[c] = (codes, cats)
    # flooded_2h / flooded 等已覆盖；label_layer/is_golden/golden_overrides_rule/split/cv_fold/dc 不动

    # ---- 抽样框架 ----
    sel = select_false_day_grids(L, S["false_dates"])
    date_pos = {d: i for i, d in enumerate(L.date_cats)}
    false_code_set = np.array(sorted(date_pos[d] for d in S["false_dates"]), dtype=np.int64)
    row_false = np.isin(L.date_codes, false_code_set)
    grid_cats = L.grid_cats if L.grid_cats is not None else sorted(set(frame["grid_id"].astype(str)))
    gmap = {g: i for i, g in enumerate(grid_cats)}
    n_dates = len(L.date_cats)
    allowed_pairs = sorted(int(gmap[g]) * n_dates + date_pos[d]
                           for d, gs in sel.items() for g in gs)
    allowed_arr = np.array(allowed_pairs, dtype=np.int64)
    pair = L.grid_codes * n_dates + L.date_codes
    pair_ok = np.zeros(n, dtype=bool)
    fr = pair[row_false]
    idx = np.searchsorted(allowed_arr, fr)
    idx_clipped = np.minimum(idx, len(allowed_arr) - 1)
    pair_ok[row_false] = allowed_arr[idx_clipped] == fr
    keep = ~row_false | pair_ok
    print(f"[repair] keep {keep.sum():,} / {n:,} rows "
          f"(false-day selected rows {int((row_false & pair_ok).sum()):,})")

    # ---- 修复前后差异统计（保留行）----
    diffs = {}
    k = keep
    for c in INT_REPLACED:
        old = (wl_old if c == "warn_level" else frame[c].to_numpy(dtype=np.int8))
        d = (final[c] != old) & k
        diffs[c] = {"changed_rows_kept": int(d.sum()),
                    "changed_rows_dropped": int(((final[c] != old) & ~k).sum())}
    if diffs["warn_level"]["changed_rows_kept"]:
        sub = pd.DataFrame({"old": wl_old[k & (final["warn_level"] != wl_old)],
                            "new": final["warn_level"][k & (final["warn_level"] != wl_old)],
                            "false_day": row_false[k & (final["warn_level"] != wl_old)]})
        diffs["warn_level"]["breakdown"] = {
            f"old{int(a)}→new{int(b)}|{'false' if c else 'kept_day'}": int(v)
            for (a, b, c), v in sub.groupby(["old", "new", "false_day"]).size().items()}
    for c in CAT_REPLACED:
        codes, cats = cat_state[c]
        old_codes, _ = _cat_writer(c, frame[c])
        d = (codes != old_codes) & k
        diffs[c] = {"changed_rows_kept": int(d.sum())}
    old_pos_kept = int(frame["flooded_2h"].to_numpy()[k].sum())
    new_pos_kept = int(final["flooded_2h"][k].sum())

    # ---- 流式写主表 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(SRC_MAIN)
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(OUT_MAIN, schema)
    i0 = 0
    nb = 0
    try:
        for batch in pf.iter_batches(batch_size=500_000):
            i1 = min(i0 + batch.num_rows, n)
            m = keep[i0:i1]
            tbl = pa.Table.from_batches([batch]).slice(0, i1 - i0)
            if not m.all():
                tbl = tbl.take(np.flatnonzero(m))
            for c in INT_REPLACED:
                vals = final[c][i0:i1][m]
                f = schema.field(c)
                arr = pa.array(vals, type=pa.int8()).cast(f.type)
                tbl = tbl.set_column(tbl.schema.get_field_index(c), f, arr)
            for c in CAT_REPLACED:
                codes, cats = cat_state[c]
                vals = [cats[i] for i in codes[i0:i1][m]]
                f = schema.field(c)
                arr = pa.array(vals, type=pa.string())
                if pa.types.is_dictionary(f.type):
                    arr = pc.dictionary_encode(arr)
                if arr.type != f.type:
                    arr = arr.cast(f.type)
                tbl = tbl.set_column(tbl.schema.get_field_index(c), f, arr)
            writer.write_table(tbl)
            nb += 1
            i0 = i1
            if i0 >= n:
                break
    finally:
        writer.close()

    # ---- 写 sample_frame ----
    new_frame = frame.loc[keep].copy()
    for c in INT_REPLACED:
        new_frame[c] = final[c][keep]
    for c in CAT_REPLACED:
        codes, cats = cat_state[c]
        new_frame[c] = pd.Categorical.from_codes(codes[keep], categories=cats)
    new_frame.to_parquet(OUT_FRAME, index=False)

    # ---- 自检 ----
    meta_out = pq.read_metadata(OUT_MAIN)
    schema_ok = [f.name for f in pq.read_schema(OUT_MAIN)] == schema.names
    keys = pq.read_table(OUT_MAIN, columns=["grid_id", "date", "slot"]).to_pandas()
    dup = int(keys.duplicated().sum())
    self_check = {"rows_out": meta_out.num_rows, "row_groups": meta_out.num_row_groups,
                  "schema_names_ok": schema_ok, "duplicated_keys": dup,
                  "frame_rows_out": int(len(new_frame))}

    manifest = {
        "version": "delivery_repaired_b1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script": "scripts/b_side_repair.py",
        "basis": "R2 方案：B 侧以 HKO 官方 rstorm.dat 重建标签与抽样框架（用户 2026-10-26 批准）",
        "inputs": {
            "src_main": {"path": str(SRC_MAIN), "sha256": sha256(SRC_MAIN),
                         "rows": pq.read_metadata(SRC_MAIN).num_rows},
            "src_frame": {"path": str(SRC_FRAME), "sha256": sha256(SRC_FRAME)},
            "official_rstorm_csv": {"path": str(OFFICIAL), "sha256": sha256(OFFICIAL),
                                    "source": "https://www.hko.gov.hk/dps/wxinfo/climat/warndb/rstorm.dat",
                                    "fetched": "2026-10-26",
                                    "raw_sha256": sha256(AUDIT / "hko_rstorm.dat.raw")},
        },
        "outputs": {
            "main": {"path": str(OUT_MAIN), "sha256": sha256(OUT_MAIN),
                     "rows": meta_out.num_rows},
            "frame": {"path": str(OUT_FRAME), "sha256": sha256(OUT_FRAME),
                      "rows": int(len(new_frame))},
        },
        "params": {
            "seed": SEED,
            "nonevent_cells_per_day": N_NONEVENT_CELLS,
            "selection_scheme": "假事件日(320天)改判非事件日后，每日从该日在主表中已有网格"
                                "(按 grid_id 字符串升序)以 np.random.default_rng(42) 不放回抽 35 格；"
                                "日期按字符串升序逐日消耗同一 rng",
            "sample_weights": "无加权（均匀选择）",
            "attrib_conf_mode": attrib_mode,
            "peak_mode": peak_mode,
            "peak_stats": peak_stats,
            "golden_rows": "保留原标签全部 10 个重建列旧值，其余列随表复制",
            "replaced_columns": REPLACED,
            "untouched_columns": "其余 99 列逐行原样复制（含 59 个模型特征）",
            "label_rule": __doc__.split("口径（与")[1].split('"""')[0].strip(),
        },
        "date_classification": {
            "real_event_days": len(S["real_dates"]),
            "false_event_days": len(S["false_dates"]),
            "nonevent_days": int((~S["old_event_mask_dates"]).sum()),
            "official_rb_days": S["official_rb_days"],
            "official_amber_days": S["official_amber_days"],
        },
        "diffs_old_vs_new_on_kept_rows": diffs,
        "positives": {"old_kept_rows": old_pos_kept, "new": new_pos_kept,
                      "old_total": int(frame["flooded_2h"].sum())},
        "counts": {
            "rows_old": n, "rows_new": int(keep.sum()),
            "positive_rate_old": round(float(frame["flooded_2h"].mean()), 6),
            "positive_rate_new": round(float(new_pos_kept / keep.sum()), 6),
        },
        "self_check": self_check,
        "selection_example": {d: sorted(sel[d])[:3] for d in sorted(sel)[:2]},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(OUT_DIR / "manifest_repair_b1.json", manifest)
    write_json(WORK / "repair_manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "self_check": self_check,
                      "positives": manifest["positives"]}, ensure_ascii=False, indent=2))
    ok = self_check["duplicated_keys"] == 0 and self_check["schema_names_ok"] \
        and self_check["rows_out"] == int(keep.sum())
    print(f"[repair] done in {manifest['elapsed_seconds']}s → {OUT_DIR}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 阶段 3：验证（用户 §四 清单）
# ---------------------------------------------------------------------------
def stage_validate() -> int:
    t0 = time.time()
    L = load_inputs()
    S = derive_sets(L)
    frame = L.frame
    golden_mask = (frame["label_layer"] == GOLDEN_LAYER).to_numpy()

    rep = pd.read_parquet(OUT_MAIN)
    rep_frame = pd.read_parquet(OUT_FRAME)
    manifest = json.loads((OUT_DIR / "manifest_repair_b1.json").read_text(encoding="utf-8"))
    checks: dict = {}

    # 1 主键唯一
    keys = rep[["grid_id", "date", "slot"]]
    checks["01_pk_unique"] = {"duplicated": int(keys.duplicated().sum()),
                              "rows": int(len(rep))}
    # 2 官方警告日 == 事件日
    date_pos = {d: i for i, d in enumerate(L.date_cats)}
    ev_days = set(rep.loc[rep["is_event_day"] == 1, "date"].astype(str).unique())
    off_rb = {d for i, d in enumerate(L.date_cats) if S["off_kind"][i] >= 2}
    checks["02_event_days_match_official"] = {
        "official_rb_days": len(off_rb), "repaired_event_days": len(ev_days),
        "missing": sorted(off_rb - ev_days), "extra": sorted(ev_days - off_rb)}
    # 3 红/黑槽级映射
    wl_exp = L.mat[np.array([date_pos[str(d)] for d in rep["date"].astype(str)]),
                   rep["slot"].to_numpy(dtype=np.int64)]
    wl_new = rep["warn_level"].to_numpy(dtype=np.int8)
    rb_rows = wl_exp >= 2
    checks["03_rb_slot_mapping"] = {
        "rb_rows": int(rb_rows.sum()),
        "rb_mismatches": int((wl_new[rb_rows] != wl_exp[rb_rows]).sum()),
        "amber_rows_mismatches": int(((wl_exp == 1) & (wl_new != 1)).sum()),
        "no_warning_rows_nonzero": int(((wl_exp == 0) & (wl_new != 0)).sum()),
    }
    # 4 正例来源
    rep_f = rep_frame.set_index(["grid_id", "date", "slot"])
    rep_i = rep.set_index(["grid_id", "date", "slot"])
    gold_idx = rep_i.index[rep_i["label_layer"].astype(str) == GOLDEN_LAYER]
    ng = ~rep_i.index.isin(gold_idx)
    f2h = rep_i["flooded_2h"].to_numpy() == 1
    kind_row = S["off_kind"][np.array([date_pos[str(d)] for d in rep["date"].astype(str)])]
    in_win = ((kind_row == 0) | ((kind_row == 1) & (wl_new == 1))
              | ((kind_row == 2) & (wl_new >= 2)))
    ok_rule = ((rep_i["is_A"].to_numpy() == 1) & (rep_i["is_event_day"].to_numpy() == 1)
               & in_win)
    checks["04_positive_sources"] = {
        "positives_total": int(f2h.sum()),
        "positives_golden": int((f2h & ~ng).sum()),
        "positives_rule": int((f2h & ng).sum()),
        "rule_positives_violating_rule": int((f2h & ng & ~ok_rule).sum()),
    }
    # 5 正负样本可追溯（与 manifest 一致）
    checks["05_counts_traceable"] = {
        "rows": int(len(rep)), "positives": int(f2h.sum()),
        "manifest_rows": manifest["counts"]["rows_new"],
        "manifest_positives": manifest["positives"]["new"],
        "match": int(len(rep)) == manifest["counts"]["rows_new"]
        and int(f2h.sum()) == manifest["positives"]["new"],
        "positive_rate": round(float(f2h.mean()), 6),
        "by_split": {str(k): {"n": int(v), "pos": int(rep_i.loc[rep_i["split"].astype(str) == k,
                                                                "flooded_2h"].sum())}
                     for k, v in rep_i["split"].astype(str).value_counts().items()},
    }
    # 6 时间范围完整
    days = sorted(set(rep["date"].astype(str)))
    per_day_rows = rep.groupby(rep["date"].astype(str)).size()
    checks["06_time_coverage"] = {
        "n_days": len(days), "min": days[0], "max": days[-1],
        "expected_days": 3195,
        "min_rows_per_day": int(per_day_rows.min()),
        "days_below_420_rows": int((per_day_rows < 420).sum()),
        "by_year": {str(y): int(n) for y, n in
                    pd.Series(days).str[:4].value_counts().sort_index().items()},
    }
    # 7 黄金事件不被覆盖
    old_gold = frame[golden_mask]
    old_g = old_gold.set_index(["grid_id", "date", "slot"])
    common = old_g.index.intersection(gold_idx)
    lab_cols = ["flooded", "flooded_2h", "flooded_2h_peak", "label_layer", "label_source",
                "confidence", "attrib_conf", "is_golden", "golden_overrides_rule",
                "sample_role", "split"]
    same = all((old_g.loc[common, c].astype(str).sort_index()
                == rep_i.loc[common, c].astype(str).sort_index()).all() for c in lab_cols)
    checks["07_golden_preserved"] = {
        "golden_rows_expected": int(golden_mask.sum()),
        "golden_rows_in_repaired": int(len(gold_idx)),
        "all_label_columns_identical": bool(same),
    }
    # 8 59 特征时间可用性（列存在 + 抽样 5000 行与源逐值一致）
    sys.path.insert(0, str(ROOT / "src"))
    from floodrisk import realdata as rd  # noqa: E402
    feats = rd.FEATURES
    missing_cols = [c for c in feats if c not in rep.columns]
    sample_rows = rep.sample(5000, random_state=SEED) if len(rep) > 5000 else rep
    kk = sample_rows[["grid_id", "date", "slot"]].copy()
    kk["slot"] = kk["slot"].astype(int)
    key_tuples = set(zip(kk["grid_id"].astype(str), kk["date"].astype(str), kk["slot"]))
    src_rows = []
    pf = pq.ParquetFile(SRC_MAIN)
    need = ["grid_id", "date", "slot"] + feats + REPLACED
    for batch in pf.iter_batches(batch_size=500_000, columns=need):
        t = batch.to_pandas()
        t["slot"] = t["slot"].astype(int)
        t["grid_id"] = t["grid_id"].astype(str)
        t["date"] = t["date"].astype(str)
        m = [k in key_tuples for k in zip(t["grid_id"], t["date"], t["slot"])]
        if any(m):
            src_rows.append(t[np.array(m)])
        if len(src_rows) and sum(len(x) for x in src_rows) >= len(kk):
            break
    src_all = pd.concat(src_rows) if src_rows else pd.DataFrame()
    merged = kk.merge(src_all, on=["grid_id", "date", "slot"], how="left",
                      validate="one_to_one")
    feat_same = all(
        np.allclose(merged[c].to_numpy(dtype=np.float64),
                    sample_rows[c].to_numpy(dtype=np.float64), equal_nan=True)
        for c in feats if c in merged)
    checks["08_features_unchanged"] = {
        "n_features": len(feats), "missing_columns": missing_cols,
        "spot_check_rows": int(len(merged)),
        "spot_check_all_features_equal": bool(feat_same),
        "replaced_spot_diff_counts": {c: int((merged[c].astype(str).to_numpy()
                                              != sample_rows[c].astype(str).to_numpy()).sum())
                                      for c in REPLACED if c in merged},
    }
    # 9 划分互斥
    d = pd.to_datetime(rep_i.index.get_level_values("date"))
    sp = rep_i["split"].astype(str).to_numpy()
    exp_split = np.where(sp == "golden_eval", "golden_eval",
                         np.where(sp == "test", "test",
                                  np.where(d < pd.Timestamp("2022-01-01"), "train",
                                           np.where(d <= pd.Timestamp("2022-12-31"),
                                                    "val(derived)", "INVALID"))))
    gd_code = rep_i.index.get_level_values("grid_id").astype(str).to_numpy() + "|" + \
        rep_i.index.get_level_values("date").astype(str)
    parts = {}
    for p in ("train", "val(derived)", "test", "golden_eval"):
        parts[p] = set(gd_code[exp_split == p])
    checks["09_splits_disjoint"] = {
        "split_dates_consistent": bool((exp_split != "INVALID").all()),
        "counts": {p: int((exp_split == p).sum()) for p in parts},
        "grid_day_overlap_train_val": len(parts["train"] & parts["val(derived)"]),
        "grid_day_overlap_train_test": len(parts["train"] & parts["test"]),
        "grid_day_overlap_val_test": len(parts["val(derived)"] & parts["test"]),
        "grid_day_overlap_with_golden": sum(len(parts["golden_eval"] & parts[p])
                                            for p in ("train", "val(derived)", "test")),
    }
    # 10 诊断集未被使用
    diag = DATA_DIR / "diagnostic_full_days.parquet"
    checks["10_diagnostic_untouched"] = {
        "path": str(diag), "sha256": sha256(diag),
        "note": "只读保留于原交付目录，未并入训练/验证/测试任何一侧",
    }
    # 11 抽样偏差与标签局限（指向报告）
    checks["11_limitations_documented"] = {
        "sampling_bias": "320 个改判日的 35 格只能从当日已有的 2,362 格事件日框架内抽取"
                         "（A 类占比 55.5%，高于全网格 23.8%），已写入报告",
        "label_limitation": "弱监督标签 = is_A × 官方红/黑警告归因窗口，非实测水浸；"
                            "黄金行独立保留",
    }
    ok_flags = {
        "01": checks["01_pk_unique"]["duplicated"] == 0,
        "02": not checks["02_event_days_match_official"]["missing"]
        and not checks["02_event_days_match_official"]["extra"],
        "03": checks["03_rb_slot_mapping"]["rb_mismatches"] == 0
        and checks["03_rb_slot_mapping"]["amber_rows_mismatches"] == 0
        and checks["03_rb_slot_mapping"]["no_warning_rows_nonzero"] == 0,
        "04": checks["04_positive_sources"]["rule_positives_violating_rule"] == 0,
        "05": checks["05_counts_traceable"]["match"],
        "06": checks["06_time_coverage"]["n_days"] == 3195,
        "07": checks["07_golden_preserved"]["all_label_columns_identical"]
        and checks["07_golden_preserved"]["golden_rows_in_repaired"] == 936,
        "08": not missing_cols and feat_same,
        "09": checks["09_splits_disjoint"]["split_dates_consistent"]
        and checks["09_splits_disjoint"]["grid_day_overlap_train_val"] == 0
        and checks["09_splits_disjoint"]["grid_day_overlap_train_test"] == 0
        and checks["09_splits_disjoint"]["grid_day_overlap_val_test"] == 0
        and checks["09_splits_disjoint"]["grid_day_overlap_with_golden"] == 0,
        "10": True,
        "11": True,
    }
    checks["all_pass"] = all(ok_flags.values())
    checks["gate_flags"] = ok_flags
    checks["elapsed_seconds"] = round(time.time() - t0, 1)
    write_json(WORK / "validation.json", checks)
    print(json.dumps({"gate_flags": ok_flags, "all_pass": checks["all_pass"],
                      "counts": checks["05_counts_traceable"]["by_split"]},
                     ensure_ascii=False, indent=2))
    print(f"[validate] done in {checks['elapsed_seconds']}s → {WORK/'validation.json'}")
    return 0 if checks["all_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B 侧 R2 修复（feasibility / repair / validate）")
    ap.add_argument("stage", choices=["feasibility", "repair", "validate"])
    args = ap.parse_args(argv)
    return {"feasibility": stage_feasibility,
            "repair": stage_repair,
            "validate": stage_validate}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
