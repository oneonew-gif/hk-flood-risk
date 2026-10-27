"""独立预测入口：只依赖已保存的 bundle + 特征表，不读训练代码、不重训。

这一步对应任务书第九阶段的最小可用形态：任何持有 ``*.joblib`` 的人都能
在不了解训练细节的情况下复现"某时刻某网格的风险分数"。

用法：
    .venv/bin/python scripts/predict_from_bundle.py \
        --bundle outputs/small_experiment_xxx/bundles/xgb.joblib \
        --input  outputs/small_experiment_xxx/test_rows.parquet \
        --out    outputs/small_experiment_xxx/test_preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from floodrisk import models as M


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"输入文件不存在：{p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    raise SystemExit(f"不支持的输入格式：{p.suffix}（仅 parquet / csv）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="加载 bundle 并对特征表打分")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-cols", default="grid_id,date,slot,timestamp")
    args = ap.parse_args(argv)

    bundle = M.load_bundle(args.bundle)
    rows = read_any(args.input)
    score = M.predict_with_bundle(bundle, rows)

    id_cols = [c for c in args.id_cols.split(",") if c in rows.columns]
    out = rows[id_cols].copy()
    out["risk_score"] = score
    out["model_backend"] = bundle.backend
    out["model_version"] = bundle.meta.get("version", "")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    print(
        json.dumps(
            {
                "rows": int(len(out)),
                "backend": bundle.backend,
                "n_features": len(bundle.features),
                "score_mean": round(float(score.mean()), 6),
                "out": str(dest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
