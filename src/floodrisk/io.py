"""数据读入、列映射与校验。

三条设计底线
------------
1. **不静默填充**：缺失值、越界值只报告不修补。你怎么处理，由你决定。
2. **不假设字段齐全**：可选特征缺失只降级，不中止；必需字段缺失才中止。
3. **合成数据必须可识别**：读取时检查同目录的 ``*.provenance.json`` 旁挂文件，
   把 ``SYNTHETIC`` 标记一路带进 Dataset、metrics 与 manifest，避免哪天把测试
   指标误写进报告。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .schema import (
    CANONICAL,
    FEATURE_FIELDS,
    FORECAST_FIELDS,
    VULNERABILITY_FIELDS,
    ResolvedColumns,
    required_fields,
    resolve_columns,
)

PROVENANCE_SYNTHETIC = "SYNTHETIC"
PROVENANCE_REAL = "REAL"


class DatasetError(RuntimeError):
    """数据不满足最低训练条件。"""


@dataclass
class Dataset:
    df: pd.DataFrame
    features: list[str]
    target: str
    provenance: str
    columns: ResolvedColumns
    report: dict[str, Any] = field(default_factory=dict)
    features_dropped: dict[str, str] = field(default_factory=dict)
    forecast_columns: list[str] = field(default_factory=list)
    """预报类列（如天文潮）：不直接入模，经 features.forecast 窗口聚合后使用"""

    # ---- 常用属性 ----
    @property
    def is_synthetic(self) -> bool:
        return self.provenance == PROVENANCE_SYNTHETIC

    # ---- 常用属性 ----
    @property
    def y(self) -> np.ndarray:
        if self.target not in self.df.columns:
            raise AttributeError(
                "Dataset 未载入标签列；预测场景无标签，不应访问 .y"
            )
        return self.df[self.target].to_numpy()

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def grid_id_col(self) -> str:
        return "grid_id"

    @property
    def timestamp_col(self) -> str | None:
        return "timestamp" if "timestamp" in self.df.columns else None

    @property
    def block_col(self) -> str | None:
        return "spatial_block_id" if "spatial_block_id" in self.df.columns else None

    def meta(self) -> dict[str, Any]:
        label_rate = None
        if self.target in self.df.columns and len(self.df):
            label_rate = float(self.df[self.target].mean())
        return {
            "n_rows": int(len(self.df)),
            "n_features": len(self.features),
            "features": list(self.features),
            "target": self.target,
            "provenance": self.provenance,
            "label_rate": label_rate,
        }

    def write_report(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.report, fh, ensure_ascii=False, indent=2, default=str)
        return p


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def detect_provenance(path: Path, override: str | None = None) -> str:
    """判定数据来源。优先级：配置覆盖 > 旁挂文件 > 文件名启发 > 未知。"""
    if override:
        return str(override).upper()
    sidecar = path.with_suffix(".provenance.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return str(payload.get("provenance", "")).upper() or "UNKNOWN"
        except (json.JSONDecodeError, OSError):
            pass
    if "synthetic" in path.name.lower() or "synthetic" in str(path.parent).lower():
        return PROVENANCE_SYNTHETIC
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# 读表
# ---------------------------------------------------------------------------
def read_table(path: str | Path, fmt: str = "csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"数据文件不存在：{p}")
    fmt = fmt.lower()
    if fmt == "csv":
        return pd.read_csv(p)
    if fmt == "parquet":
        try:
            return pd.read_parquet(p)
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise DatasetError("读取 parquet 需要 pyarrow：pip install pyarrow") from exc
    if fmt == "xlsx":
        return pd.read_excel(p)
    raise DatasetError(f"不支持的数据格式：{fmt}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def load_dataset(
    cfg: Config, path: str | Path | None = None, *, require_label: bool = True
) -> Dataset:
    mode = cfg.mode
    data_path = Path(path or cfg.get("data.input_path"))
    df = read_table(data_path, cfg.get("data.format", "csv"))
    if df.empty:
        raise DatasetError(f"数据为空：{data_path}")

    provenance = detect_provenance(data_path, cfg.get("project.provenance_override", None))
    issues: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append({"level": level, "code": code, "message": message})
        if level == "error" and cfg.get("data.strict", False):
            raise DatasetError(message)

    # ---- 1. 列名解析
    resolved = resolve_columns(
        df.columns,
        cfg.get("data.columns", {}) or {},
        mode=mode,
    )
    for canonical, reason in resolved.unresolved.items():
        level = "error" if canonical in required_fields(mode) else "info"
        add(level, "unresolved_field", f"字段 {canonical}: {reason}")
    for note in resolved.notes:
        add("info", "column_note", note)
    if resolved.extra:
        add(
            "info",
            "extra_columns",
            f"{len(resolved.extra)} 列未被契约识别，已忽略：{resolved.extra[:12]}",
        )

    missing_required = [f for f in required_fields(mode, require_label) if not resolved.has(f)]
    if missing_required:
        raise DatasetError(
            f"缺少必需字段 {missing_required}（mode={mode}）。"
            f"若数据里的列名不同，请在 configs 的 data.columns 中显式映射，"
            f"例如 columns: {{grid_id: YOUR_COL, flood_label: YOUR_LABEL}}。"
            f"当前数据列：{list(df.columns)[:20]}"
        )

    # ---- 2. 重命名为规范名，避免下游到处处理别名
    rename = {actual: canonical for canonical, actual in resolved.mapping.items()}
    df = df.rename(columns=rename)

    # ---- 3. 类型处理
    grid_col = "grid_id"
    df[grid_col] = df[grid_col].astype(str).str.strip()
    empty_ids = int((df[grid_col] == "").sum())
    if empty_ids:
        add("error", "empty_grid_id", f"{empty_ids} 行 grid_id 为空")

    ts_col = resolved.mapping.get("timestamp")
    if mode == "dynamic" and ts_col:
        before_na = df["timestamp"].isna().sum()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        na_ts = int(df["timestamp"].isna().sum())
        if na_ts:
            raise DatasetError(
                f"timestamp 解析失败 {na_ts} 行（原始缺失 {before_na} 行）。"
                f"dynamic 模式必须能按时间切分，请先清洗时间列"
            )

    # 数值型字段强制转换，记录转换失败量（观测类 + 预报类 + 脆弱类）
    numeric_fields = [
        c for c in FEATURE_FIELDS + FORECAST_FIELDS + VULNERABILITY_FIELDS if c in df.columns
    ]
    coerce_stats: dict[str, int] = {}
    for col in numeric_fields:
        original_na = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        gained = int(df[col].isna().sum()) - original_na
        if gained > 0:
            coerce_stats[col] = gained
            add("warning", "coerce_failed", f"{col}: {gained} 个非数值被置为 NaN")

    # ---- 4. 标签
    target = cfg.get("task.target", "flood_label")
    if require_label:
        if target not in df.columns:
            raise DatasetError(f"目标列 {target} 不存在；可选字段里也没有匹配的标签列")
        df[target] = df[target].map(_to_binary)
        bad_label = int(df[target].isna().sum())
        if bad_label:
            raise DatasetError(
                f"标签列 {target} 有 {bad_label} 行既不是 0 也不是 1（含 NaN）。"
                f"标签口径必须由你确定，本框架不代劳推断"
            )
        df[target] = df[target].astype(int)

        n_pos = int((df[target] == 1).sum())
        n_neg = int((df[target] == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise DatasetError(f"标签只有单一类别（正例 {n_pos} / 负例 {n_neg}），无法训练二分类模型")
    else:
        # 预测模式：标签可选，缺失不中止
        if target in df.columns:
            df[target] = df[target].map(_to_binary)

    # ---- 5. 重复与完整性
    # 政策：默认不删除任何数据。区分两类问题：
    #   (a) 完全重复行 —— 所有列取值一致，去掉不损失信息，但也要显式开启才删
    #   (b) 主键相同但字段冲突 —— 框架无法判断孰对孰错，一律中止
    key_cols = [grid_col] + (["timestamp"] if mode == "dynamic" else [])
    dup = int(df.duplicated(subset=key_cols).sum())
    dropped_dups = 0
    if dup:
        dup_key_mask = df.duplicated(subset=key_cols, keep=False)
        exact_mask = df.duplicated(keep=False)
        exact_dups = int((dup_key_mask & exact_mask).sum())
        conflict_mask = dup_key_mask & ~exact_mask
        conflict_rows = int(conflict_mask.sum())
        if conflict_rows:
            keys = df.loc[conflict_mask, key_cols].drop_duplicates()
            preview = keys.head(5).to_dict("records")
            raise DatasetError(
                f"{len(keys)} 个主键对应了不同取值（{conflict_rows} 行），例如 {preview}。"
                f"同一 {key_cols} 出现冲突记录时，框架不判断哪条正确，必须由人裁定后再入库"
            )
        if cfg.get("data.drop_duplicates", False):
            df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
            dropped_dups = exact_dups
            add("warning", "duplicates", f"{exact_dups} 行完全重复（{key_cols}），按配置已保留首条")
        else:
            raise DatasetError(
                f"{exact_dups} 行完全重复（键 {key_cols}）。默认不删除真实数据；"
                "确认去重安全后显式设置 data.drop_duplicates: true"
            )

    # ---- 6. 逐列质量画像
    per_column: dict[str, dict[str, Any]] = {}
    for col in numeric_fields:
        spec = CANONICAL[col]
        series = df[col]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_rate": round(float(series.isna().mean()), 6),
        }
        valid = series.dropna()
        if len(valid):
            info.update(
                min=float(valid.min()),
                max=float(valid.max()),
                mean=round(float(valid.mean()), 6),
            )
            if spec.valid_range is not None and len(valid):
                lo, hi = spec.valid_range
                oor = int(((valid < lo) | (valid > hi)).sum())
                info["out_of_range"] = oor
                info["valid_range"] = [lo, hi]
                info["unit"] = spec.unit
                if oor:
                    add(
                        "warning",
                        "out_of_range",
                        f"{col}: {oor} 个值超出合理区间 [{lo}, {hi}] {spec.unit or ''}，"
                        f"实际范围 [{valid.min():.3f}, {valid.max():.3f}]",
                    )
        missing_rate = float(series.isna().mean())
        if missing_rate > 0.3:
            add(
                "warning",
                "high_missing",
                f"{col}: 缺失率 {missing_rate:.1%}，建议在建模前明确插补或剔除策略",
            )
        per_column[col] = info

    # ---- 7. 特征清单（观测类直接入模；预报类单独收集，批 2 做窗口聚合）
    features, dropped, forecast_cols = _select_features(df, cfg)
    if not features:
        raise DatasetError(
            "没有任何可用特征。请检查 data.columns 映射，或确认数据包含 "
            f"{list(FEATURE_FIELDS)} 中的至少一个"
        )
    fc_cfg = cfg.get("features.forecast", {}) or {}
    if fc_cfg.get("enabled", False):
        missing_fc = [v for v in (fc_cfg.get("variables") or []) if v not in forecast_cols]
        if missing_fc:
            add(
                "warning",
                "forecast_missing",
                f"features.forecast.variables 中 {missing_fc} 不在数据里，预报类特征将被跳过",
            )

    # ---- 8. 规模统计
    n_grids = int(df[grid_col].nunique())
    stats: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_grids": n_grids,
        "n_features": len(features),
    }
    if mode == "dynamic":
        stats["n_timestamps"] = int(df["timestamp"].nunique())
        span = df["timestamp"].max() - df["timestamp"].min()
        stats["time_span"] = str(span)
        stats["time_min"] = str(df["timestamp"].min())
        stats["time_max"] = str(df["timestamp"].max())

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_path": str(data_path),
        "provenance": provenance,
        "mode": mode,
        "stats": stats,
        "columns_mapping": dict(resolved.mapping),
        "unresolved_fields": dict(resolved.unresolved),
        "extra_columns": resolved.extra,
        "features_used": features,
        "features_dropped": dropped,
        "forecast_columns": forecast_cols,
        "label": (
            {
                "positive": n_pos,
                "negative": n_neg,
                "positive_rate": round(n_pos / (n_pos + n_neg), 6),
            }
            if require_label
            else {"note": "预测模式：无标签，正负样本统计未计算"}
        ),
        "duplicates": {"found": dup, "dropped": dropped_dups},
        "coerce_failed": coerce_stats,
        "per_column": per_column,
        "issues": issues,
        "notes": [
            "本报告只做诊断，不做任何修补；缺失值/越界值如何处理由使用者决定。",
        ],
    }
    if provenance == PROVENANCE_SYNTHETIC:
        report["notes"].append(
            "数据来源标记为 SYNTHETIC：所有指标、概率与地图仅用于程序自测，禁止写入报告或用于答辩。"
        )
    elif provenance != PROVENANCE_REAL:
        report["notes"].append(
            "数据来源未能确认（UNKNOWN）：默认不可引用。若确为真实数据，"
            "请旁挂 *.provenance.json（{\"provenance\": \"REAL\"}），"
            "或在 configs 中设置 project.provenance_override: REAL"
        )

    return Dataset(
        df=df,
        features=features,
        target=target,
        provenance=provenance,
        columns=resolved,
        report=report,
        features_dropped=dropped,
        forecast_columns=forecast_cols,
    )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _to_binary(value: Any) -> float:
    """把标签规整成 0/1；无法判定则返回 NaN。"""
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "1.0", "true", "yes", "y", "是"}:
            return 1.0
        if text in {"0", "0.0", "false", "no", "n", "否"}:
            return 0.0
        return np.nan
    try:
        num = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(num):
        return np.nan
    if num in (0.0, 1.0):
        return num
    return np.nan


def _select_features(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[list[str], dict[str, str], list[str]]:
    """决定哪些列进入模型，并记录每个被剔除字段的原因。

    返回 (观测类特征, 剔除记录, 预报类列)。
    预报类列（天文潮等）不直接入模：单点值意义有限，真正有信息量的是
    [t, t+horizon) 窗口上的聚合（如窗口最大潮位），那是 features.py（批 2）的工作。
    """
    dropped: dict[str, str] = {}
    candidates: list[str] = [c for c in FEATURE_FIELDS if c in df.columns]
    forecast_cols: list[str] = [c for c in FORECAST_FIELDS if c in df.columns]
    for col in forecast_cols:
        dropped[col] = (
            "预报类变量：发起时刻已知的未来值（如天文潮）。不直接入模，"
            "由 features.forecast 对 [t, t+horizon) 窗口聚合后使用（批 2）"
        )

    vuln_cols = [c for c in VULNERABILITY_FIELDS if c in df.columns]
    vuln_in_model = bool(cfg.get("features.vulnerability.in_model", False))
    if vuln_in_model:
        candidates.extend(vuln_cols)
    else:
        for col in vuln_cols:
            dropped[col] = (
                "脆弱性字段默认不进模型（features.vulnerability.in_model=false）："
                "人口密度不是水浸的物理成因，放进模型会让特征重要性无法解释。"
                "如需纳入请显式打开开关，manifest 会记录该决定"
            )

    include = [c for c in (cfg.get("features.include", []) or []) if c]
    exclude = {c for c in (cfg.get("features.exclude", []) or []) if c}

    if include:
        unknown = [c for c in include if c not in df.columns]
        if unknown:
            raise DatasetError(
                f"features.include 中的列在数据里不存在：{unknown}。"
                f"请先确认列映射，或从 include 中移除"
            )
        candidates = [c for c in include if c in candidates or c in df.columns]
        if vuln_cols and not vuln_in_model:
            candidates = [c for c in candidates if c not in vuln_cols]
        candidates = [c for c in candidates if c not in forecast_cols]

    selected: list[str] = []
    for col in candidates:
        if col in exclude:
            dropped[col] = "被 features.exclude 显式剔除"
            continue
        if col == cfg.get("task.target"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            dropped[col] = f"非数值类型（{df[col].dtype}），暂不支持自动编码"
            continue
        if df[col].notna().sum() == 0:
            dropped[col] = "全为缺失值"
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped[col] = "取值单一，无区分度"
            continue
        selected.append(col)

    # 顺序稳定，便于结果对比
    order = {name: i for i, name in enumerate(FEATURE_FIELDS + VULNERABILITY_FIELDS)}
    selected.sort(key=lambda c: order.get(c, 999))
    return selected, dropped, forecast_cols
