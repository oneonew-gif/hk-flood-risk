"""GIS 导出模块：把预测概率写到 GeoJSON，供 GIS 软件或前端地图加载。

为什么需要这一层
----------------
训练/预测产出的是表格（CSV），GIS 可视化需要带几何的 GeoJSON。
本模块只做"表格 → 带几何的 GeoJSON"的映射，不重新计算概率、不做分级。

输入边界
--------
- 预测结果：``predict()`` 的 DataFrame 或 CSV 路径，必须含 ``grid_id``，
  dynamic 模式必须含 ``timestamp``；
- 网格几何：GeoJSON FeatureCollection，每个 Feature 的 properties 含
  ``grid_id``，geometry 为 Polygon；
- 两者通过 ``grid_id`` 关联，找不到 geometry 的网格被跳过并告警。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class GISExportError(RuntimeError):
    """GIS 导出条件不成立。"""


def _load_grid_layer(path: str | Path) -> dict[str, Any]:
    """加载网格 GeoJSON，返回 {grid_id: geometry} 映射。"""
    p = Path(path)
    if not p.exists():
        raise GISExportError(f"grid_layer 不存在：{p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GISExportError(f"grid_layer 不是合法 JSON：{p}，{exc}")

    if payload.get("type") != "FeatureCollection":
        raise GISExportError(f"grid_layer 必须是 FeatureCollection，实际为 {payload.get('type')!r}")

    mapping: dict[str, Any] = {}
    for feat in payload.get("features", []):
        props = feat.get("properties", {})
        gid = props.get("grid_id")
        if gid is None:
            continue
        mapping[str(gid)] = feat.get("geometry")
    if not mapping:
        raise GISExportError("grid_layer 中没有任何含 grid_id 的 Feature")
    return mapping


def export_geojson(
    predictions: pd.DataFrame | str | Path,
    grid_layer: str | Path,
    output_path: str | Path,
    *,
    crs: str = "EPSG:4326",
    fields: list[str] | None = None,
    provenance: str = "UNKNOWN",
    run_id: str | None = None,
) -> Path:
    """把预测概率导出为 GeoJSON。

    Parameters
    ----------
    predictions
        ``predict()`` 返回的 DataFrame，或 CSV 路径。
        必须含 ``grid_id``；dynamic 模式建议含 ``timestamp``。
    grid_layer
        网格 GeoJSON 路径（FeatureCollection，properties 含 ``grid_id``）。
    output_path
        输出 GeoJSON 路径。
    crs
        坐标系，默认 ``EPSG:4326``。写入输出文件的 ``crs`` 字段（非标准，
        但部分 GIS 工具会读取）。
    fields
        保留的 properties 字段列表。为 None 时默认保留：
        ``["grid_id", "timestamp", "prob_calibrated", "provenance", "run_id"]``
        （实际存在的列才写入）。
    provenance
        数据来源标记，写入每个 Feature 的 properties。
    run_id
        训练运行 ID，写入每个 Feature 的 properties（可选）。

    Returns
    -------
    Path
        输出 GeoJSON 路径。

    Notes
    -----
    - 找不到 geometry 的 ``grid_id`` 会被跳过，不中止；
    - 若 ``predictions`` 含 ``timestamp``，会转成 ISO 格式字符串；
    - 不重新计算概率，只做映射。
    """
    # ---- 1. 加载预测结果
    if isinstance(predictions, (str, Path)):
        preds = pd.read_csv(predictions)
    else:
        preds = predictions.copy()

    if "grid_id" not in preds.columns:
        raise GISExportError("predictions 缺少 grid_id 列")

    # ---- 2. 加载网格几何
    geom_map = _load_grid_layer(grid_layer)

    # ---- 3. 确定保留字段
    if fields is None:
        default_fields = ["grid_id", "timestamp", "prob_calibrated"]
        fields = [f for f in default_fields if f in preds.columns]
    else:
        fields = [f for f in fields if f in preds.columns]

    # ---- 4. 组装 Feature
    features = []
    skipped = 0
    for row in preds.itertuples(index=False):
        gid = str(getattr(row, "grid_id"))
        geom = geom_map.get(gid)
        if geom is None:
            skipped += 1
            continue

        props: dict[str, Any] = {}
        for f in fields:
            val = getattr(row, f, None)
            if f == "timestamp" and pd.notna(val):
                props[f] = pd.Timestamp(val).isoformat()
            else:
                props[f] = val
        # provenance / run_id 是导出参数，始终写入（不受 fields 过滤）
        props["provenance"] = provenance
        if run_id is not None:
            props["run_id"] = run_id

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })

    # ---- 5. 写出
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "provenance": provenance,
            "run_id": run_id,
            "n_input_rows": int(len(preds)),
            "n_exported": len(features),
            "n_skipped_no_geometry": skipped,
        },
        "features": features,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


__all__ = ["GISExportError", "export_geojson"]
