"""字段契约与数据校验测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from floodrisk.io import DatasetError, detect_provenance, load_dataset, read_table
from floodrisk.schema import required_fields, resolve_columns


# ---------------------------------------------------------------- schema
def test_alias_resolution():
    cols = ["Grid", "Rain_mm", "tide_level", "Label"]
    res = resolve_columns(cols)
    assert res.mapping["grid_id"] == "Grid"
    assert res.mapping["rainfall"] == "Rain_mm"
    assert res.mapping["tide"] == "tide_level"
    assert res.mapping["flood_label"] == "Label"


def test_explicit_override_wins_over_alias():
    cols = ["grid", "rain_total", "rf"]
    res = resolve_columns(cols, overrides={"rainfall": "rain_total"})
    assert res.mapping["rainfall"] == "rain_total"
    # 未被覆盖的 grid 仍按别名匹配
    assert res.mapping["grid_id"] == "grid"


def test_override_pointing_to_missing_column_is_reported():
    res = resolve_columns(["grid", "label"], overrides={"rainfall": "nope"})
    assert "rainfall" in res.unresolved
    assert "nope" in res.unresolved["rainfall"]


def test_required_fields_by_mode():
    assert set(required_fields("dynamic")) == {"grid_id", "timestamp", "flood_label"}
    assert set(required_fields("static")) == {"grid_id", "flood_label"}


# ---------------------------------------------------------------- io
def test_dataset_dynamic_basics(dataset_dynamic):
    ds = dataset_dynamic
    assert ds.provenance == "SYNTHETIC"
    assert ds.is_synthetic
    assert "grid_id" in ds.df.columns
    assert ds.target == "flood_label"
    assert set(ds.features) >= {"rainfall", "tide", "wind_speed", "elevation"}
    assert len(ds.features) == len(set(ds.features))
    assert ds.report["stats"]["n_rows"] == len(ds.df)


def test_vulnerability_excluded_by_default(dataset_dynamic):
    ds = dataset_dynamic
    for col in ("population", "vulnerability_score"):
        assert col not in ds.features
        assert col in ds.features_dropped
        assert "不是水浸的物理成因" in ds.features_dropped[col]


def test_vulnerability_can_be_enabled(cfg_dynamic):
    cfg_dynamic._data["features"]["vulnerability"]["in_model"] = True
    ds = load_dataset(cfg_dynamic)
    assert "vulnerability_score" in ds.features


def test_dirty_data_is_reported_not_silently_fixed(cfg_dynamic, project_root, tmp_path):
    from floodrisk.synthetic import write_synthetic_bundle

    bundle = write_synthetic_bundle(tmp_path / "dirty", mode="dynamic", n_grids=16, n_steps=48, seed=3, dirty=True)
    cfg = cfg_dynamic.__class__.load(
        project_root / "configs" / "synthetic_dynamic.yaml",
        overrides={
            "data": {"input_path": str(bundle["data"]), "drop_duplicates": True},
            "tuning": {"enabled": False},
        },
    )
    ds = load_dataset(cfg)
    codes = {i["code"] for i in ds.report["issues"]}
    assert "out_of_range" in codes       # tide 被注入 -3.5
    assert "duplicates" in codes         # 注入了重复行
    assert ds.report["duplicates"]["dropped"] > 0
    # 缺失值没有被填：rainfall 仍有 NaN
    assert ds.df["rainfall"].isna().any()


def _panel_with_dup(cfg_dynamic, tmp_path, *, conflict: bool):
    df = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "timestamp": ["2025-01-01 00:00", "2025-01-01 01:00"] * 2,
            "rainfall": [1.0, 2.0, 3.0, 4.0],
            "flood_label": [0, 1, 0, 1],
        }
    )
    extra = df.iloc[[0]].copy()
    if conflict:
        extra["rainfall"] = 99.0   # 同主键、字段冲突
    df = pd.concat([df, extra], ignore_index=True)
    path = tmp_path / "dup_panel.csv"
    df.to_csv(path, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(path)
    return cfg_dynamic


def test_exact_duplicates_abort_by_default(cfg_dynamic, tmp_path):
    """默认不删除真实数据：完全重复行必须中止，而不是悄悄留首条。"""
    cfg = _panel_with_dup(cfg_dynamic, tmp_path, conflict=False)
    with pytest.raises(DatasetError, match="完全重复"):
        load_dataset(cfg)


def test_conflicting_duplicate_keys_abort_even_when_drop_enabled(cfg_dynamic, tmp_path):
    """同主键不同取值：框架不替人裁定，即使开启去重也必须中止。"""
    cfg = _panel_with_dup(cfg_dynamic, tmp_path, conflict=True)
    cfg._data["data"]["drop_duplicates"] = True
    with pytest.raises(DatasetError, match="主键对应了不同取值"):
        load_dataset(cfg)


def test_missing_required_field_raises(cfg_dynamic, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"grid_id": ["a", "b"], "timestamp": ["2025-01-01", "2025-01-02"]}).to_csv(bad, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(bad)
    with pytest.raises(DatasetError, match="必需字段"):
        load_dataset(cfg_dynamic)


def test_non_binary_label_raises(cfg_dynamic, tmp_path):
    bad = tmp_path / "bad_label.csv"
    pd.DataFrame(
        {
            "grid_id": ["a", "b", "c"],
            "timestamp": ["2025-01-01"] * 3,
            "rainfall": [1.0, 2.0, 3.0],
            "flood_label": [0, 1, 2],
        }
    ).to_csv(bad, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(bad)
    with pytest.raises(DatasetError, match="标签"):
        load_dataset(cfg_dynamic)


def test_single_class_label_raises(cfg_dynamic, tmp_path):
    bad = tmp_path / "single.csv"
    pd.DataFrame(
        {
            "grid_id": list("abcdef"),
            "timestamp": ["2025-01-01"] * 6,
            "rainfall": range(6),
            "flood_label": [0] * 6,
        }
    ).to_csv(bad, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(bad)
    with pytest.raises(DatasetError, match="单一类别"):
        load_dataset(cfg_dynamic)


def test_report_roundtrip(dataset_dynamic, tmp_path):
    p = dataset_dynamic.write_report(tmp_path / "validation_report.json")
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["provenance"] == "SYNTHETIC"
    assert "features_used" in payload
    assert any("SYNTHETIC" in n for n in payload["notes"])


def test_provenance_sidecar_detection(dyn_bundle):
    assert detect_provenance(dyn_bundle["data"]) == "SYNTHETIC"
    assert detect_provenance(dyn_bundle["data"], override="real") == "REAL"


def test_unknown_provenance_is_flagged_not_trusted(cfg_dynamic, tmp_path):
    """未旁挂 provenance 的数据必须按 UNKNOWN 处理：报告明确"默认不可引用"。"""
    df = pd.DataFrame(
        {
            "grid_id": ["a", "a", "b", "b"],
            "timestamp": ["2025-01-01 00:00", "2025-01-01 01:00"] * 2,
            "rainfall": [1.0, 2.0, 3.0, 4.0],
            "flood_label": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "observed_panel.csv"  # 文件名不含 synthetic，也无旁挂文件
    df.to_csv(path, index=False)
    cfg_dynamic._data["data"]["input_path"] = str(path)
    cfg_dynamic._data["project"]["provenance_override"] = None  # 合成配置里固定了 SYNTHETIC，放开它
    ds = load_dataset(cfg_dynamic)
    assert ds.provenance == "UNKNOWN"
    assert any("UNKNOWN" in n and "不可引用" in n for n in ds.report["notes"])


def test_static_dataset(dataset_static):
    ds = dataset_static
    assert ds.timestamp_col is None
    assert ds.df["grid_id"].is_unique
    assert "elevation" in ds.features


def test_read_table_missing_file_raises():
    with pytest.raises(DatasetError, match="不存在"):
        read_table("/tmp/definitely-not-here-xyz.csv")
