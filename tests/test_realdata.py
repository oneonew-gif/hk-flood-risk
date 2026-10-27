"""A 同学交付数据集适配层的回归测试。

重点守两件事：
1. 目标位移**只在同网格、槽位连续**时发生 —— 跨网格、跨缺口一律 NaN；
2. 主模型特征清单与排除清单不相交，规则/标签字段不可能"手滑"进模。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodrisk import realdata as rd


def _row(grid: str, date: str, slot: int, label: int = 0, *, split: str = "train") -> dict:
    return {
        "grid_id": grid,
        "date": date,
        "slot": slot,
        "split": split,
        "is_event_day": 1,
        "is_A": 1,
        "warn_level": 3 if label else 0,
        "flooded_2h": label,
    }


def _panel() -> pd.DataFrame:
    rows: list[dict] = []
    # G1：两天连续，slot 11 的下一槽是次日 slot 0
    for slot in range(12):
        rows.append(_row("G1", "2022-01-01", slot, label=int(slot == 11)))
    for slot in range(12):
        rows.append(_row("G1", "2022-01-02", slot, label=int(slot == 0)))
    # G2：只有一天 → 末槽没有下一槽，必须为 NaN（绝不能接到 G3 的第一行）
    for slot in range(12):
        rows.append(_row("G2", "2022-01-01", slot, label=1))
    # G3：槽位有缺口（0,1,3）→ slot 1 的下一槽不是 slot 2，必须为 NaN
    for slot in (0, 1, 3):
        rows.append(_row("G3", "2022-01-01", slot, label=0))
    return pd.DataFrame(rows)


def test_primary_target_is_the_current_slot_without_shift():
    out, _ = rd.build_targets(_panel())
    assert out["y_future_2h"].tolist() == out["flooded_2h"].tolist()
    # 原始标签列必须原样保留（不覆盖 A 的数据）
    assert "flooded_2h" in out.columns


def test_shift_target_crosses_day_within_same_grid():
    out, _ = rd.build_targets(_panel())
    g1 = out[out["grid_id"].eq("G1")].sort_values(["date", "slot"])
    day1_last = g1[(g1["date"].eq("2022-01-01")) & (g1["slot"].eq(11))].iloc[0]
    day2_first = g1[(g1["date"].eq("2022-01-02")) & (g1["slot"].eq(0))].iloc[0]
    assert day1_last["y_future_2h_shift"] == day2_first["flooded_2h"]


def test_shift_target_never_takes_next_grid_row():
    out, _ = rd.build_targets(_panel())
    g2 = out[out["grid_id"].eq("G2")]
    last = g2[g2["slot"].eq(11)].iloc[0]
    assert np.isnan(last["y_future_2h_shift"])  # G3 的行就在后面，也不许取


def test_shift_target_never_jumps_over_slot_gap():
    out, _ = rd.build_targets(_panel())
    g3 = out[out["grid_id"].eq("G3")].sort_values("slot")
    slot1 = g3[g3["slot"].eq(1)].iloc[0]
    assert np.isnan(slot1["y_future_2h_shift"])
    assert g3[g3["slot"].eq(0)]["y_future_2h_shift"].iloc[0] == g3[g3["slot"].eq(1)]["flooded_2h"].iloc[0]


def test_target_report_counts_shift_holes():
    _, report = rd.build_targets(_panel())
    # 4 处无位移标签：G1 末槽（无次日）、G2 末槽、G3 缺口处 slot 1、G3 末槽
    assert report["rows_without_shift_target"] == 4
    assert report["n_grids"] == 3


def test_assign_splits_and_partition():
    df = pd.DataFrame(
        [
            _row("G1", "2021-12-31", 0, split="train"),
            _row("G1", "2022-06-01", 0, split="train"),
            _row("G1", "2022-06-02", 0, split="train"),
            _row("G1", "2023-05-01", 0, split="test"),
            _row("G1", "2018-09-16", 0, split="golden_eval"),
        ]
    )
    df["date"] = pd.Categorical(df["date"])
    df["y_future_2h"] = df["flooded_2h"].astype("int8")
    out, info = rd.assign_splits(df)
    mapping = dict(zip(out["date"].astype(str), out["exp_split"].astype(str)))
    assert mapping["2021-12-31"] == "train"
    assert mapping["2022-06-01"] == "val"
    assert mapping["2023-05-01"] == "test"
    assert mapping["2018-09-16"] == "golden_eval"  # 标记保留，但不进 train/val/test
    assert info["definition"]["test"] == "A.split==test"

    part = rd.check_partition(out)
    assert part["grid_day_overlap_train_test"] == 0
    assert part["date_overlap_train_test"] == 0


def test_feature_registry_is_disjoint_from_exclusions():
    feats, excluded = set(rd.FEATURES), set(rd.EXCLUDED)
    assert feats
    assert not (feats & excluded)
    assert len(rd.FEATURES) == len(feats)  # 无重复
    for col in ("flooded_2h", "warn_level", "is_A", "warn_level_day", "is_event_day", "split"):
        assert col in excluded
    assert rd.RULE_COLUMNS and not (set(rd.RULE_COLUMNS) & feats)


def test_no_in_slot_future_column_enters_model():
    # 槽内降雨 / 增水 是 t=槽开始时尚未发生的量，必须在排除清单里
    for prefix in ("power_rain_slot_2h", "era5_rain_slot_2h", "surge_max"):
        assert any(c.startswith(prefix) or c == f"{prefix}_m" for c in rd.EXCLUDED)


def test_load_delivery_adds_slot_start_timestamp(tmp_path):
    df = _panel().assign(**{c: 0.0 for c in rd.FEATURES})
    for c in ("grid_id", "date"):
        df[c] = df[c].astype(str)
    path = tmp_path / "panel.parquet"
    df.to_parquet(path, index=False)

    out = rd.load_delivery(str(path), columns=["grid_id", "date", "slot", "flooded_2h"],
                            categorical=["grid_id", "date"])
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    ts = out.loc[out["slot"].eq(5), "timestamp"].iloc[0]
    assert ts.hour == 10  # slot 5 → 10:00，锚点是槽开始
    assert out["timestamp"].notna().all()


def test_rule_baseline_reproduces_label():
    df = _panel()
    pred = rd.rule_baseline(df)
    assert (pred == df["flooded_2h"].to_numpy()).all()


def test_stratified_sample_covers_year_and_event_flag():
    df = _panel()
    df["exp_split"] = pd.Categorical(["train"] * len(df))
    df["date_dt"] = pd.to_datetime(df["date"])
    df["is_event_day"] = np.where(df["date"].eq("2022-01-02"), 0, 1)
    s = rd.stratified_sample(df, 4)
    assert set(s["is_event_day"].unique()) == {0, 1}
    assert set(s["date"].astype(str).str[:4]) == {"2022"}


def test_build_targets_requires_label_column():
    df = _panel().drop(columns=["flooded_2h"])
    with pytest.raises(ValueError, match="缺少构造目标所需列"):
        rd.build_targets(df)
