"""A 同学交付数据集 v10 的适配层：读入、目标对齐、特征清单、切分。

为什么单独一层
--------------
交付宽表是 ``(grid_id, date, slot)`` 三键、109 列、混合时间锚的宽表；
现有 ``schema/io/features`` 假设"单一 timestamp + 14 个 canonical 特征 + 自动 lag"，
直接套用会（a）缺 timestamp 报错，（b）逐网格重建 2.11 亿行时间轴 OOM，
（c）把 ERA5 与 POWER 两套降雨当成两组独立特征重复 lag。

本模块只做三件事，不训练、不出图：
1. ``load_delivery``  读入并合成 timestamp（时间锚 = 槽开始）；
2. ``build_targets``  构造未来 2 小时目标，严格按 grid 分组、校验槽位连续性；
3. ``feature_registry`` 给出「主模型特征 / 排除字段 + 理由」的唯一定义。

时间锚的实测依据（见 reports/02_prediction_target.md）：
    prev6h/12h/24h/48h 与"截至本槽开始"的累加 MAE = 0.000，
    与"含本槽"的累加 MAE = 0.37~0.56 → 这些列是 **槽开始锚定**；
    rain_slot_2h / surge_max / warn_level / flooded_2h 描述的是**槽内**（未来）。
因此一行的合法预测时刻是槽开始 t = date 00:00 + 2*slot 小时，
标签 ``flooded_2h(slot)`` 恰好是 (t, t+2h] 内是否水浸，**无需位移**。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["grid_id", "date", "slot"]

# ---------------------------------------------------------------------------
# 排除字段（唯一来源，主模型特征不得与之相交）
# ---------------------------------------------------------------------------
EXCLUDED: dict[str, str] = {
    # 标识 / 键
    "grid_id": "主键",
    "date": "主键",
    "slot": "主键",
    "tide_station": "标识（最近潮位站）",
    "hydro1km_grid_id": "标识（1 km 父格）",
    # 标签族
    "flooded": "日级父标签",
    "flooded_2h": "当期标签（目标本身）",
    "flooded_2h_peak": "标签族",
    # 标签元数据
    "label_layer": "编码标签生成过程",
    "label_source": "编码标签生成过程",
    "confidence": "编码标签生成过程",
    "attrib_conf": "编码标签生成过程",
    "is_golden": "抽样/黄金标记",
    "golden_overrides_rule": "抽样/黄金标记",
    "sample_role": "抽样角色",
    # 划分
    "split": "A 的切分列",
    "cv_fold": "A 的 18 区留一折",
    "dc": "空间分组键（不作特征）",
    # 警告 / 规则内部：标签由此生成
    "warn_level": "标签生成器：flooded_2h 的归因窗口 = 警告活跃槽",
    "warn_level_day": "日级警告：含当天未来信息，且与 is_event_day 等价",
    "is_event_day": "日级事件日标志：含当天未来信息（warn_level_day>=2 的等价编码）",
    "is_A": "A 类空间规则输出（标签的另一半）",
    "is_A_provisional": "A 类规则临时值",
    "A_reason": "A 类规则触发条件分解",
    "dist_to_blackspot_m": "A 类规则输入（DSD 黑点先验）",
    "dist_to_lowlying_m": "A 类规则输入（低洼易发区先验）",
    "flood_prone_flag": "A 类规则输入",
    "coastal_lowlying_flag": "A 类规则输入",
    "lowland_flag_500": "A 类规则输入",
    "hydro1km_lowland_flag": "A 类规则输入",
    "low_coastal_flag": "A 类规则输入",
    "dist_to_landedge_m": "A 类规则输入（高程<=5m 且 距陆缘<=500m），按「参与标签生成」从严剔除",
    # 时间锚不符：槽内观测 = 未来信息
    "era5_rain_slot_2h_mm": "槽内降雨：t=槽开始时尚未发生",
    "power_rain_slot_2h_mm": "槽内降雨：t=槽开始时尚未发生",
    "surge_max_m": "槽内最大增水（未来量）且训练期 2018-2021 全缺",
    # 结构性说明：era5 prev/api72 8 列在 v10.0 曾因「2018/2019/2021/2022 全缺」列入排除；
    # v10.1 补齐后（2018-2025 NaN=0%）移出排除、进入 ERA5 特征组（方案 A，见 FEATURE_GROUPS）。
    # 人口 / 社会：只作分级叠加，不进概率模型
    "pop_worldpop": "社会承灾体，约定不进模型",
    "pop_density_gis": "社会承灾体，约定不进模型",
    "pop_worldpop_pixels": "社会承灾体",
    "pop_density_dcd": "社会承灾体（区级常量）",
    "elderly_frac_gis": "社会脆弱性，约定不进模型",
    "median_income_gis": "社会脆弱性，约定不进模型",
    # 坐标：让模型逐格记忆 A 类掩码
    "i": "坐标：可逐格记忆标签空间掩码",
    "j": "坐标",
    "cx_2326": "坐标",
    "cy_2326": "坐标",
    "lon": "坐标",
    "lat": "坐标",
    # 常量
    "esa_lc_70_snow_and_ice_frac": "全 0 常量",
    "esa_lc_100_moss_and_lichen_frac": "全 0 常量",
}

# ---------------------------------------------------------------------------
# 主模型特征（全部为 t=槽开始时已知，或静态不变量）
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, list[str]] = {
    "降雨·前期累计（POWER，截至 t）": [
        "power_rain_prev3h_mm",
        "power_rain_prev6h_mm",
        "power_rain_prev12h_mm",
        "power_rain_prev24h_mm",
        "power_rain_prev48h_mm",
        "power_rain_max1h_prev6h_mm",
        "power_rain_max1h_prev24h_mm",
        "power_api72_mm",
    ],
    "潮汐（天文潮为确定性预报，t 时刻可知未来槽）": [
        "tide_pred_max_m",
        "tide_pred_mean_m",
        "tide_n_hours",
        "dist_to_tide_m",
    ],
    "地形（静态）": [
        "elevation",
        "elevation_min",
        "elevation_max",
        "elevation_std",
        "slope_deg",
        "elv_rel_mean",
        "tpi_500",
        "roughness_500",
        "land_frac",
        "dem_n",
    ],
    "地表覆盖 / 不透水面（静态）": [
        "esa_lc_10_tree_cover_frac",
        "esa_lc_20_shrubland_frac",
        "esa_lc_30_grassland_frac",
        "esa_lc_40_cropland_frac",
        "esa_lc_50_built_up_frac",
        "esa_lc_60_bare_sparse_vegetation_frac",
        "esa_lc_80_permanent_water_bodies_frac",
        "esa_lc_90_herbaceous_wetland_frac",
        "esa_lc_95_mangroves_frac",
        "esa_lc_pixels",
        "esa_valid_frac",
        "impervious_frac",
        "lu_built_frac",
        "lu_green_frac",
        "lu_water_frac",
        "lu_valid_frac",
        "landuse_class",
    ],
    "排水设施距离（静态）": ["dist_to_pump", "dist_to_outfall"],
    "1 km 水文派生（静态）": [
        "hydro1km_hand_mean",
        "hydro1km_hand_min",
        "hydro1km_hand_p10",
        "hydro1km_twi_mean",
        "hydro1km_flow_acc_km2",
        "hydro1km_relief_std",
        "hydro1km_elv_rel_mean",
    ],
    "时间循环": ["month", "hour_sin", "hour_cos"],
}

# v10.0 特征方案快照（51 列）：v10.1 引入 ERA5 之前的方案，固化用于新旧对照实验。
POWER_ONLY_FEATURES: list[str] = [c for g in FEATURE_GROUPS.values() for c in g]

# v10.1 方案 A（用户 2026-10-24 批准）：ERA5 降雨 8 列与 POWER 并列，保留来源标识、不合并。
# v10.1 已补齐 ERA5（2018-2025 NaN=0%）；时间锚实测为「截至槽开始」，与 POWER 一致。
FEATURE_GROUPS["降雨·前期累计（ERA5，截至 t）"] = [
    "era5_rain_prev3h_mm",
    "era5_rain_prev6h_mm",
    "era5_rain_prev12h_mm",
    "era5_rain_prev24h_mm",
    "era5_rain_prev48h_mm",
    "era5_rain_max1h_prev6h_mm",
    "era5_rain_max1h_prev24h_mm",
    "era5_api72_mm",
]

FEATURES: list[str] = [c for g in FEATURE_GROUPS.values() for c in g]

# 降雨族（含各源 api72 前期指数）与时间循环：特征族分解用。
# 注意：v10.0 实验曾用 startswith("power_rain") 划族，把 power_api72_mm 误归静态族
# （static_only 实为 41 列）。本清单为修正口径，新旧对照时已在报告中注明。
RAIN_FEATURES: list[str] = [
    "power_rain_prev3h_mm", "power_rain_prev6h_mm", "power_rain_prev12h_mm",
    "power_rain_prev24h_mm", "power_rain_prev48h_mm", "power_rain_max1h_prev6h_mm",
    "power_rain_max1h_prev24h_mm", "power_api72_mm",
    "era5_rain_prev3h_mm", "era5_rain_prev6h_mm", "era5_rain_prev12h_mm",
    "era5_rain_prev24h_mm", "era5_rain_prev48h_mm", "era5_rain_max1h_prev6h_mm",
    "era5_rain_max1h_prev24h_mm", "era5_api72_mm",
]
TIME_FEATURES: list[str] = ["month", "hour_sin", "hour_cos"]

# 独立规则基线（仅用于对照，绝不入模）
RULE_COLUMNS = ["is_A", "warn_level"]


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------
def load_delivery(
    path: str,
    *,
    columns: list[str] | None = None,
    filter_keys: pd.DataFrame | None = None,
    row_groups: int | None = None,
    categorical: list[str] | None = None,
) -> pd.DataFrame:
    """读交付宽表，补 ``timestamp``（= 槽开始）。

    参数
    ----
    columns     要读取的列（None = 全部）
    filter_keys 若给定（含 grid_id/date/slot 三列），只保留命中这些键的行；
                用于"先算目标、再只取样本特征"，避免把 5-6 GB 全表载入 16 GB 机器
    row_groups  仅调试用，限制读取的 row group 数
    categorical 需要低内存表示的列（如 grid_id/date）；在 arrow 层做字典编码，
                再转 pandas Categorical。1184 万行的对象字符串列会吃掉数 GB，
                这是个纯工程约束，不改变任何取值。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_groups = pf.metadata.num_row_groups
    if row_groups is not None:
        n_groups = min(n_groups, row_groups)

    if filter_keys is None:
        tbl = pq.read_table(path, columns=columns)
        return _add_timestamp(_to_pandas(tbl, categorical))

    keys = filter_keys.copy()
    keys["slot"] = keys["slot"].astype(np.int64)
    key_tuples = set(zip(keys["grid_id"].astype(str), keys["date"].astype(str), keys["slot"]))

    parts: list = []  # arrow Table 片段，最后一次性 concat 再转 pandas
    for gi in range(n_groups):
        t = pf.read_row_group(gi, columns=columns)
        g = t.column("grid_id").to_pylist()
        d = t.column("date").to_pylist()
        s = t.column("slot").to_pylist()
        idx = [i for i, k in enumerate(zip(g, d, s)) if k in key_tuples]
        if idx:
            parts.append(t.take(pa.array(idx)))
    if not parts:
        raise ValueError("filter_keys 未命中任何行")
    return _add_timestamp(_to_pandas(pa.concat_tables(parts), categorical))


def _to_pandas(tbl, categories: list[str] | None) -> pd.DataFrame:
    """arrow → pandas；指定列做字典编码后再转 Categorical，压住对象字符串内存。"""
    if not categories:
        return tbl.to_pandas()
    import pyarrow.compute as pc

    for name in categories:
        if name not in tbl.column_names:
            continue
        idx = tbl.schema.get_field_index(name)
        col = tbl.column(idx)
        if not (hasattr(col.type, "value_type") and str(col.type).startswith("dictionary")):
            col = pc.dictionary_encode(col)
        tbl = tbl.set_column(idx, name, col)
    return tbl.to_pandas(categories=list(categories))


def _add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """时间锚 = 槽开始：t = date 00:00 + 2*slot 小时。"""
    if isinstance(df["date"].dtype, pd.CategoricalDtype):
        date_str = df["date"].astype(object).astype(str)
    else:
        date_str = df["date"].astype(str)
    ts = pd.to_datetime(date_str, format="%Y-%m-%d") + pd.to_timedelta(
        df["slot"].astype(int) * 2, unit="h"
    )
    df = df.copy()
    df.insert(3, "timestamp", ts)
    return df


# ---------------------------------------------------------------------------
# 目标构造
# ---------------------------------------------------------------------------
def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """构造两个目标，并返回对齐报告。

    主目标 ``y_future_2h``（槽开始锚定）
        t = date 00:00 + 2*slot；y = flooded_2h(slot)。
        flooded_2h(slot) 描述 [2s, 2s+2) 内是否水浸，恰为 (t, t+2h]。**不做位移**。

    对照目标 ``y_future_2h_shift``（槽结束锚定，即"前移 1 槽"）
        t = 槽结束；y = 下一槽的 flooded_2h。只在"同网格 且 槽位连续"时成立：
        同日 slot s → s+1，或 slot 11 → 次日 slot 0（次日必须存在）。
        跨网格、跨缺失槽一律置 NaN，绝不按行位移。

    原始 ``flooded_2h`` 保留不动，仅新增两列。
    """
    need = {"grid_id", "date", "slot", "flooded_2h"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"缺少构造目标所需列：{sorted(missing)}")

    out = df.copy()
    date_str = out["date"].astype(object).astype(str) if isinstance(
        out["date"].dtype, pd.CategoricalDtype
    ) else out["date"].astype(str)
    out["date_dt"] = pd.to_datetime(date_str, format="%Y-%m-%d")
    out = out.sort_values(["grid_id", "date_dt", "slot"], kind="mergesort").reset_index(drop=True)

    # 主目标：槽内标签本身就是未来 2 小时
    out["y_future_2h"] = out["flooded_2h"].astype("int8")

    g = out.groupby("grid_id", sort=False)
    nxt_slot = g["slot"].shift(-1)
    nxt_date = g["date_dt"].shift(-1)
    nxt_label = g["flooded_2h"].shift(-1)

    same_day = (nxt_slot == out["slot"] + 1) & (nxt_date == out["date_dt"])
    cross_day = (
        (out["slot"] == 11)
        & (nxt_slot == 0)
        & (nxt_date == out["date_dt"] + pd.Timedelta(days=1))
    )
    valid = same_day | cross_day
    out["y_future_2h_shift"] = np.where(valid, nxt_label, np.nan)

    # ---- 连续性诊断
    per_grid_day = out.groupby(["grid_id", "date_dt"], observed=True).size()
    slot_gap = out.groupby("grid_id", observed=True)["slot"].apply(
        lambda s: int((s.diff() > 1).sum())
    )

    report = {
        "n_rows": int(len(out)),
        "n_grids": int(out["grid_id"].nunique()),
        "n_grid_days": int(len(per_grid_day)),
        "grid_day_slot_count": {int(k): int(v) for k, v in per_grid_day.value_counts().sort_index().items()},
        "grid_days_with_12_slots": int((per_grid_day == 12).sum()),
        "grid_days_with_less": int((per_grid_day < 12).sum()),
        "rows_with_shift_target": int(valid.sum()),
        "rows_without_shift_target": int((~valid).sum()),
        "shift_missing_same_day_last_slot": int(((out["slot"] == 11) & ~cross_day).sum()),
        "shift_missing_grid_end": int(g["slot"].shift(-1).isna().sum()),
        "slot_gaps_total": int(slot_gap.sum()),
        "grids_with_slot_gaps": int((slot_gap > 0).sum()),
        "target_positive_rate": float(out["y_future_2h"].mean()),
        "shift_target_positive_rate": float(out["y_future_2h_shift"].mean(skipna=True)),
        "note": "y_future_2h 直接在原行上取 flooded_2h；y_future_2h_shift 仅在网格内槽位连续时取值",
    }
    return out, report


def alignment_examples(df: pd.DataFrame, grid_id: str | None = None, date: str | None = None,
                       n_slots: int = 14) -> pd.DataFrame:
    """打印用：给出某个网格-日及其跨日的连续槽，展示 t、标签窗口、位移结果。"""
    d = df
    if grid_id is not None:
        d = d[d["grid_id"] == grid_id]
    if date is not None:
        d = d[d["date"] == date]
    d = d.sort_values(["grid_id", "date", "slot"]).head(n_slots)
    keep = ["grid_id", "date", "slot", "timestamp", "y_future_2h", "y_future_2h_shift",
            "power_rain_prev3h_mm", "tide_pred_max_m"]
    keep = [c for c in keep if c in d.columns]
    r = d[keep].copy()
    r["预测时刻_t"] = r["timestamp"]
    r["预测窗口"] = r["timestamp"].dt.strftime("%H:%M") + "→" + (
        r["timestamp"] + pd.Timedelta(hours=2)
    ).dt.strftime("%H:%M")
    return r.drop(columns=["timestamp"])


# ---------------------------------------------------------------------------
# 切分（provided 的真实定义）
# ---------------------------------------------------------------------------
def assign_splits(
    df: pd.DataFrame, *, val_start: str = "2022-01-01", val_end: str = "2022-12-31"
) -> tuple[pd.DataFrame, dict]:
    """按 A 的 ``split`` 列切分，并在训练期内划出验证段。

    provided 的实际定义（写进报告，不许默认成立）：
    - ``test``   = A 的 split=="test"（2023-01-01 ~ 2026-09-30）
    - ``golden_eval`` = A 的 split=="golden_eval"，本次实验**不参与训练与主评估**
    - ``train``  = A 的 split=="train" 且 date <  val_start
    - ``val``    = A 的 split=="train" 且 val_start <= date <= val_end
    标签窗口在单槽内闭合，因此 train/val/test 之间**不需要 purge**。
    """
    d = df.copy()
    date_str = d["date"].astype(object).astype(str) if isinstance(
        d["date"].dtype, pd.CategoricalDtype
    ) else d["date"].astype(str)
    d["date_dt"] = pd.to_datetime(date_str, format="%Y-%m-%d")
    a_split = d["split"].astype(str)

    d["exp_split"] = "excluded"
    d.loc[a_split.eq("test"), "exp_split"] = "test"
    d.loc[a_split.eq("golden_eval"), "exp_split"] = "golden_eval"
    train_period = a_split.eq("train")
    d.loc[train_period & d["date_dt"].lt(pd.Timestamp(val_start)), "exp_split"] = "train"
    d.loc[
        train_period
        & d["date_dt"].ge(pd.Timestamp(val_start))
        & d["date_dt"].le(pd.Timestamp(val_end)),
        "exp_split",
    ] = "val"
    d["exp_split"] = d["exp_split"].astype("category")

    info = {
        "definition": {
            "test": "A.split==test",
            "golden_eval": "A.split==golden_eval（本次不参与训练/主评估）",
            "train": f"A.split==train 且 date < {val_start}",
            "val": f"A.split==train 且 {val_start} <= date <= {val_end}",
        },
        "no_purge_reason": "标签 flooded_2h 描述单槽内部，窗口不跨槽，故无需 purge",
        "counts": {str(k): int(v) for k, v in d["exp_split"].value_counts().items()},
        "date_ranges": {
            str(k): [str(v["date_dt"].min().date()), str(v["date_dt"].max().date())]
            for k, v in d.groupby("exp_split", observed=True)
        },
    }
    return d, info


def _codes(s: pd.Series) -> np.ndarray:
    """列 → int64 编码；Categorical 直接取 codes，避免再造一列对象字符串。"""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.to_numpy().astype(np.int64)
    return pd.factorize(s)[0].astype(np.int64)


def check_partition(df: pd.DataFrame) -> dict:
    """切分自检：时间不重叠、网格-日不跨段（同一 grid-day 不得同时落在两段）。

    1184 万行用 set(zip(...)) 会把机器撑爆，这里把 (grid_id, date) 编码成整数
    再取集合交集 —— 等价判定，内存只与唯一 grid-day 数成正比。
    """
    gid_code = _codes(df["grid_id"])
    date_code = _codes(df["date"])
    gd = gid_code * (int(date_code.max()) + 1) + date_code

    mask = {k: df["exp_split"].eq(k).to_numpy() for k in ("train", "val", "test")}
    gd_sets = {k: np.unique(gd[m]) for k, m in mask.items()}
    out = {
        "n": {k: int(m.sum()) for k, m in mask.items()},
        "pos_rate": {
            k: float(df.loc[m, "y_future_2h"].mean()) for k, m in mask.items()
        },
        "unique_grid_days": {k: int(len(v)) for k, v in gd_sets.items()},
        "grid_day_overlap_train_val": int(len(np.intersect1d(gd_sets["train"], gd_sets["val"]))),
        "grid_day_overlap_train_test": int(len(np.intersect1d(gd_sets["train"], gd_sets["test"]))),
        "grid_day_overlap_val_test": int(len(np.intersect1d(gd_sets["val"], gd_sets["test"]))),
        "date_overlap_train_test": int(
            len(np.intersect1d(np.unique(date_code[mask["train"]]), np.unique(date_code[mask["test"]])))
        ),
    }
    return out


# ---------------------------------------------------------------------------
# 独立规则基线（对照用，不入模）
# ---------------------------------------------------------------------------
def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """A 的弱监督规则本身：is_A × [warn_level>=2]。仅作上限对照。"""
    need = {"is_A", "warn_level"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"规则基线需要列 {sorted(missing)}")
    return ((df["is_A"].astype(int) == 1) & (df["warn_level"].astype(int) >= 2)).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# 抽样
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n: int, *, seed: int = 42) -> pd.DataFrame:
    """按 (年 × 事件日) 分层抽样，保证年份、事件日/非事件日都有覆盖。

    非事件日只占全表 10%，按行数比例分会几乎采不到，故给每层设下限。
    """
    d = df[df["exp_split"].isin(["train", "val", "test"])].copy()
    if "date_dt" in d.columns:
        year_int = d["date_dt"].dt.year.astype("int32")
    else:
        year_int = d["date"].astype(str).str[:4].astype("int32")
    # 整数层号（年*10 + 事件日），避免为 1184 万行拼字符串
    d["stratum"] = year_int * 10 + d["is_event_day"].astype("int32")
    d["year"] = year_int.astype(str)
    frac = n / len(d)
    parts = []
    for _, g in d.groupby("stratum", sort=True):
        take = max(int(round(len(g) * frac)), min(len(g), 1500))
        parts.append(g.sample(min(take, len(g)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out
