#!/usr/bin/env python3
"""网格历史评分数据准备（只读转换：outputs/ → hko_api/data/latest/grid_scores.json）。

隔离保证（2026-10-26 用户指令）：
  - 输入只读：outputs/formal_training_repaired_b1_20261026-020814/ 下的
    predictions_test.parquet 与 gis/grid_risk_500m.csv，不写入、不移动、不修改；
  - 输出仅写到 hko_api/data/latest/grid_scores.json（实时接入独立目录）；
  - 不调用模型、不重算预测、不改动任何论文正式实验产物。

口径标注（照实写入 meta，不得混称）：
  - test_mean = 测试期（2023-01-01 ~ 2026-09-30）全部采样槽逐格平均 p_dual_xgb，
    弱监督标签下探索性风险评分，非实时水浸发生概率；
  - all_mean / rank = 正式训练 GIS 产物（train+val+test 全期逐格平均，非黄金标准口径）。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent / "outputs" / "formal_training_repaired_b1_20261026-020814"
LATEST = ROOT / "data" / "latest"

DISCLAIMER = ("网格评分为历史模型输出（弱监督标签下探索性风险排序），"
              "不是实时水浸发生概率；与实时气象数据分图层展示。")

MAX_AGE_MIN = None  # 静态历史评分，无过期概念（由 meta 注明）


def main() -> None:
    # ---- 只读输入 ----
    pred = pd.read_parquet(FORMAL / "predictions_test.parquet",
                           columns=["grid_id", "p_dual_xgb"])
    g = pred.groupby("grid_id", observed=True)["p_dual_xgb"].agg(["mean", "count"])
    g = g.reset_index().rename(columns={"mean": "test_mean", "count": "n_test"})
    g["grid_id"] = g["grid_id"].astype(str)

    gis = pd.read_csv(FORMAL / "gis" / "grid_risk_500m.csv")  # 只读
    df = gis[["grid_id", "lon_wgs84", "lat_wgs84", "mean_risk_all",
              "rank_mean_risk"]].merge(g, on="grid_id", how="left")

    grids = [
        {"id": r.grid_id,
         "lon": round(float(r.lon_wgs84), 5),
         "lat": round(float(r.lat_wgs84), 5),
         "test_mean": None if pd.isna(r.test_mean) else round(float(r.test_mean), 5),
         "all_mean": round(float(r.mean_risk_all), 5),
         "all_rank": int(r.rank_mean_risk),
         "n_test": None if pd.isna(r.n_test) else int(r.n_test)}
        for r in df.itertuples(index=False)
    ]

    out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kind": "HISTORICAL_MODEL_SCORES",  # 与实时气象 snapshot（kind=REALTIME_HKO）严格区分
        "model": "dual_xgb（正式训练 delivery_repaired_b1，未修改）",
        "layers": {
            "test_mean": {
                "label": "历史评分·测试期（2023-01~2026-09）",
                "definition": "测试期全部采样槽逐格平均 p_dual_xgb；弱监督标签下探索性风险评分",
            },
            "all_mean": {
                "label": "历史评分·全期（train+val+test）",
                "definition": "正式训练 GIS 产物 mean_risk_all；全期逐格平均，非黄金标准口径",
            },
        },
        "n_grids": len(grids),
        "disclaimer": DISCLAIMER,
        "grids": grids,
    }
    LATEST.mkdir(parents=True, exist_ok=True)
    (LATEST / "grid_scores.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (LATEST / "grid_scores.js").write_text(
        "window.HKO_GRID_SCORES = " + json.dumps(out, ensure_ascii=False) + ";",
        encoding="utf-8")
    print(f"grid_scores.json written: {len(grids)} grids "
          f"(test_mean non-null: {sum(1 for x in grids if x['test_mean'] is not None)})")


if __name__ == "__main__":
    main()
