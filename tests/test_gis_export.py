"""gis_export.py 测试：GeoJSON 导出格式、字段保留、异常处理。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from floodrisk.gis_export import GISExportError, export_geojson


@pytest.fixture
def grid_layer(tmp_path):
    """构造一个简化的 grid_layer GeoJSON（4 个网格）。"""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "properties": {"grid_id": "G0000", "spatial_block_id": "Z00"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]],
                },
                "properties": {"grid_id": "G0001", "spatial_block_id": "Z00"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 1], [1, 1], [1, 2], [0, 2], [0, 1]]],
                },
                "properties": {"grid_id": "G0002", "spatial_block_id": "Z01"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
                },
                "properties": {"grid_id": "G0003", "spatial_block_id": "Z01"},
            },
        ],
    }
    p = tmp_path / "grid_layer.geojson"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "grid_id": ["G0000", "G0001", "G0002"],
            "timestamp": ["2024-01-01 00:00:00"] * 3,
            "prob_raw": [0.1, 0.5, 0.9],
            "prob_calibrated": [0.12, 0.48, 0.85],
        }
    )


def test_export_basic_geojson(grid_layer, predictions, tmp_path):
    out = tmp_path / "predictions.geojson"
    result = export_geojson(predictions, grid_layer, out, provenance="SYNTHETIC")

    assert result.exists()
    geo = json.loads(result.read_text(encoding="utf-8"))
    assert geo["type"] == "FeatureCollection"
    assert geo["crs"]["properties"]["name"] == "EPSG:4326"
    assert len(geo["features"]) == 3
    assert geo["metadata"]["n_input_rows"] == 3
    assert geo["metadata"]["n_exported"] == 3
    assert geo["metadata"]["n_skipped_no_geometry"] == 0
    assert geo["metadata"]["provenance"] == "SYNTHETIC"

    props = geo["features"][0]["properties"]
    assert props["grid_id"] == "G0000"
    assert props["prob_calibrated"] == 0.12
    assert props["provenance"] == "SYNTHETIC"
    assert "timestamp" in props


def test_export_preserves_fields(grid_layer, predictions, tmp_path):
    out = tmp_path / "predictions.geojson"
    export_geojson(
        predictions,
        grid_layer,
        out,
        fields=["grid_id", "prob_calibrated"],
        provenance="REAL",
        run_id="test_run_001",
    )
    geo = json.loads(out.read_text(encoding="utf-8"))
    props = geo["features"][0]["properties"]
    assert props["grid_id"] == "G0000"
    assert props["prob_calibrated"] == 0.12
    assert "timestamp" not in props  # 未列入 fields
    assert props["run_id"] == "test_run_001"


def test_export_missing_grid_id_raises(grid_layer, tmp_path):
    bad = pd.DataFrame({"x": [1], "prob_calibrated": [0.5]})
    with pytest.raises(GISExportError, match="grid_id"):
        export_geojson(bad, grid_layer, tmp_path / "out.geojson")


def test_export_missing_grid_layer_raises(predictions, tmp_path):
    with pytest.raises(GISExportError, match="grid_layer 不存在"):
        export_geojson(predictions, "/nonexistent/path.geojson", tmp_path / "out.geojson")


def test_export_skips_unknown_grid_id(grid_layer, predictions, tmp_path):
    preds = predictions.copy()
    preds.loc[3] = ["G9999", "2024-01-01 00:00:00", 0.3, 0.35]
    out = tmp_path / "predictions.geojson"
    export_geojson(preds, grid_layer, out)
    geo = json.loads(out.read_text(encoding="utf-8"))
    assert geo["metadata"]["n_input_rows"] == 4
    assert geo["metadata"]["n_exported"] == 3
    assert geo["metadata"]["n_skipped_no_geometry"] == 1


def test_export_from_csv_path(grid_layer, predictions, tmp_path):
    csv_path = tmp_path / "preds.csv"
    predictions.to_csv(csv_path, index=False)
    out = tmp_path / "predictions.geojson"
    result = export_geojson(csv_path, grid_layer, out)
    assert result.exists()
    geo = json.loads(result.read_text(encoding="utf-8"))
    assert len(geo["features"]) == 3


def test_export_timestamp_iso_format(grid_layer, predictions, tmp_path):
    out = tmp_path / "predictions.geojson"
    export_geojson(predictions, grid_layer, out)
    geo = json.loads(out.read_text(encoding="utf-8"))
    ts = geo["features"][0]["properties"]["timestamp"]
    assert "T" in ts  # ISO 格式含 T
