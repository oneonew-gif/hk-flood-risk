"""特征构造：把"行锚定"落到具体实现上。

三个必须分别处理的特征族
------------------------
1. **观测类 · 滞后** ``{var}_lag{k}``
   定义：``X[t] = var[t - k·step]``。k 必须 >= ``Config.min_feature_lag``。
   即使 k=1、面板有缺报，也必须按**时间轴**回看，不能按行数回看 ——
   实现上先把每个网格重索引到完整时间轴，再 shift。

2. **观测类 · 滚动** ``{var}_r{w}{stat}``
   定义：先位移 ``min_feature_lag`` 步，再取该点之前 w 个完整时间步的统计量。
   等价于"窗口右端固定在过去、且窗口内必须全是真实观测"。

3. **预报类 · 未来窗口** ``{var}_now`` / ``{var}_fw{h}{stat}``
   天文潮这类确定性未来值：``X[t] = max(var[t .. t+h-1])``。
   这是**合法的未来信息** —— 它在发起时刻 t 就能算出来。
   为防止误用，只接受 ``schema.kind == "forecast"`` 的变量；把 rainfall 塞进
   ``features.forecast.variables`` 会直接报错，而不是悄悄放行。

不做的两件事
------------
- **不填缺失**：历史不足或缺口导致算不出来的地方保持 NaN，由下游按行剔除并记录数量。
- **不给静态字段造滞后**：elevation / slope / drainage 在同一网格上恒定，
  ``elevation_lag3`` 与 ``elevation`` 是同一列，只会给模型加 5 份噪声。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .io import Dataset
from .schema import FORECAST_FIELDS


class FeatureError(RuntimeError):
    """特征构造条件不成立。"""


_FLAG_PATTERNS = {
    # 这些字段名一旦出现在特征里，说明标签期信息漏进了特征侧
    "future_looking": ("_future", "_next", "_ahead"),
}


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
class FeatureSet:
    def __init__(
        self,
        X: pd.DataFrame,
        *,
        families: dict[str, list[str]],
        report: dict,
        valid_mask: np.ndarray,
    ):
        self.X = X
        self.families = families
        self.report = report
        self.valid_mask = valid_mask

    @property
    def names(self) -> list[str]:
        return list(self.X.columns)

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, idx: np.ndarray) -> pd.DataFrame:
        return self.X.iloc[idx]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_features(dataset: Dataset, cfg: Config) -> FeatureSet:
    """按配置构造特征矩阵，与 ``dataset.df`` 行序严格对齐。"""
    if cfg.mode == "static":
        return _build_static_features(dataset, cfg)

    df = dataset.df
    grid_col = dataset.grid_id_col
    ts_col = dataset.timestamp_col
    if ts_col is None:
        raise FeatureError("dynamic 模式缺少 timestamp 列，无法构造时序特征")

    step_hours = float(cfg.get("task.time_step_hours", 1.0))
    min_lag = int(cfg.min_feature_lag)
    lags = [int(k) for k in (cfg.get("features.temporal.lags", []) or [])]
    windows = [int(w) for w in (cfg.get("features.temporal.rolling_windows", []) or [])]
    stats = list(cfg.get("features.temporal.rolling_stats", []) or [])
    horizon = int(cfg.horizon_steps)

    if any(k < min_lag for k in lags):
        raise FeatureError(
            f"lags={lags} 中有小于最小合法滞后 {min_lag} 的取值；"
            f"配置校验本应拦住它，说明配置对象被绕过构造了"
        )

    obs_cols = [c for c in dataset.features]
    dyn_cols, static_cols, undecidable = _split_dynamic_static(df, grid_col, obs_cols)

    forecast_vars = _resolve_forecast_vars(dataset, cfg)
    fc_stats = list(cfg.get("features.forecast.window_stats", ["max"]) or ["max"])

    frames: list[pd.DataFrame] = []
    for gid, orig in df.groupby(grid_col, sort=False):
        axis = _full_axis(orig, ts_col, step_hours)
        out = pd.DataFrame(index=axis.index)

        # ---- 静态：整网格一个值，直接广播
        for col in static_cols:
            out[col] = float(orig[col].iloc[0])

        # ---- 观测类 · 当前值（仅当 min_lag == 0 才合法）
        if min_lag == 0:
            for col in dyn_cols:
                out[col] = axis[col]

        # ---- 观测类 · 滞后
        for col in dyn_cols:
            for k in lags:
                out[f"{col}_lag{k}"] = axis[col].shift(k)

        # ---- 观测类 · 滚动（先位移，再滚 —— 顺序不能反）
        for col in dyn_cols:
            for w in windows:
                base = axis[col].shift(min_lag).rolling(w)
                for st in stats:
                    out[f"{col}_r{w}{st}"] = getattr(base, st)()

        # ---- 预报类 · 未来窗口
        for var in forecast_vars:
            out[f"{var}_now"] = axis[var]
            for w in (horizon,):
                for st in fc_stats:
                    out[f"{var}_fw{w}{st}"] = _future_window(axis[var], w, st)

        # 回填到原始行位置（缺报行的计算结果丢弃）
        out = out.reindex(orig[ts_col].to_numpy())
        out.index = orig.index
        frames.append(out)

    X = pd.concat(frames).reindex(df.index)

    families = _families(X.columns, dyn_cols, static_cols, forecast_vars, lags, windows, stats, horizon)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    usable = [c for c in X.columns if c not in all_nan]
    X = X[usable]
    families = {k: [c for c in v if c in usable] for k, v in families.items()}

    valid_mask = X.notna().all(axis=1).to_numpy()
    report = {
        "n_rows": int(len(X)),
        "n_features": int(len(X.columns)),
        "min_feature_lag": min_lag,
        "horizon_steps": horizon,
        # lags / rolling_windows 的单位是「步」，实际小时数 = 步数 × step_hours
        "step_hours": step_hours,
        "temporal_units": "steps",
        "dynamic_vars": dyn_cols,
        "static_vars": static_cols,
        "forecast_vars": forecast_vars,
        "dropped_all_nan": all_nan,
        "undecidable_vars": sorted(undecidable),
        "families": {k: len(v) for k, v in families.items()},
        "usable_rows": int(valid_mask.sum()),
        "unusable_rows": int((~valid_mask).sum()),
        "unusable_reason": (
            "历史不足或序列缺口导致部分特征为 NaN；框架不填补，直接按行剔除。"
            "剔除量会写进 manifest"
        ),
    }
    if not usable:
        raise FeatureError(
            "没有任何特征被成功构造。请检查 features.temporal 配置与数据的时间跨度"
        )
    return FeatureSet(X, families=families, report=report, valid_mask=valid_mask)


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------
def _build_static_features(dataset: Dataset, cfg: Config) -> FeatureSet:
    """静态易发性：一网格一行，直接用原始字段，**禁止任何时序派生**。

    静态模式下 time_conventions / lags / forecast 一律不生效 —— 若配置里留了
    lags，``Config.validate`` 已经先一步拦住。这里不做第二层猜测。
    """
    cols = list(dataset.features)
    if not cols:
        raise FeatureError("静态模式没有可用特征，请检查 data.columns 映射")
    X = dataset.df[cols].copy()
    valid_mask = X.notna().all(axis=1).to_numpy()
    report = {
        "mode": "static",
        "n_rows": int(len(X)),
        "n_features": len(cols),
        "families": {"static": len(cols)},
        "usable_rows": int(valid_mask.sum()),
        "unusable_rows": int((~valid_mask).sum()),
        "note": "静态模式不做时序派生；缺失行不填补，直接标记为不可用",
    }
    return FeatureSet(
        X, families={"static": cols}, report=report, valid_mask=valid_mask
    )


def _split_dynamic_static(
    df: pd.DataFrame, grid_col: str, cols: list[str]
) -> tuple[list[str], list[str], dict[str, str]]:
    """区分"随网格内时间变化"与"每网格恒定"的特征。

    判据是网格内 nunique(dropna) <= 1 → 静态。这条判据会让"某网格全程恒定"
    的动态变量被误判为静态，但那类列本身也没有时序信息可挖，代价可接受。
    """
    dyn, static, odd = [], [], {}
    for col in cols:
        nun = df.groupby(grid_col, sort=False)[col].nunique(dropna=True)
        if (nun > 1).any():
            dyn.append(col)
        elif (nun == 0).all():
            odd[col] = "所有网格该列均为空"
        else:
            static.append(col)
    return dyn, static, odd


def _resolve_forecast_vars(dataset: Dataset, cfg: Config) -> list[str]:
    """只接受 schema 声明为 forecast 的变量，其他一律报错。"""
    fc = cfg.get("features.forecast", {}) or {}
    if not fc.get("enabled", False):
        return []
    want = list(fc.get("variables", []) or [])
    illegal = [v for v in want if v not in FORECAST_FIELDS]
    if illegal:
        raise FeatureError(
            f"features.forecast.variables 中 {illegal} 不是预报类变量（schema.kind != forecast）。"
            f"只有'发起时刻即可精确得到的未来值'才能进未来窗口，例如天文潮 {list(FORECAST_FIELDS)}；"
            f"把实测雨量放进未来窗口等于直接把标签期信息喂给模型"
        )
    missing = [v for v in want if v not in dataset.df.columns]
    if missing:
        raise FeatureError(
            f"features.forecast.variables 中 {missing} 不在数据里。"
            f"当前数据含预报类列：{dataset.forecast_columns or '无'}"
        )
    return want


def _full_axis(orig: pd.DataFrame, ts_col: str, step_hours: float) -> pd.DataFrame:
    """把某网格重索引到完整时间轴（缺口保留为 NaN）。"""
    sub = orig.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    if sub.index.duplicated().any():
        raise FeatureError("存在重复 (grid_id, timestamp)，请先按 configs.data.drop_duplicates 去重")
    full = pd.date_range(sub.index.min(), sub.index.max(), freq=pd.Timedelta(hours=step_hours))
    return sub.reindex(full)


def _future_window(series: pd.Series, width: int, stat: str) -> pd.Series:
    """``out[t] = stat(series[t .. t+width-1])`` —— 向前看 width 步。

    只允许用于确定性预报量。实现上把序列倒序、滚动、再倒回来，比写负数 shift
    更不容易看错符号。
    """
    if width < 1:
        raise FeatureError("未来窗口宽度必须 >= 1")
    rev = series.iloc[::-1]
    rolled = getattr(rev.rolling(width), stat)()
    return rolled.iloc[::-1]


def _families(
    columns,
    dyn_cols,
    static_cols,
    forecast_vars,
    lags,
    windows,
    stats,
    horizon,
) -> dict[str, list[str]]:
    lag_names = {f"{c}_lag{k}" for c in dyn_cols for k in lags}
    roll_names = {f"{c}_r{w}{s}" for c in dyn_cols for w in windows for s in stats}
    fc_now = {f"{v}_now" for v in forecast_vars}
    fc_win = {f"{v}_fw{horizon}{s}" for v in forecast_vars for s in {"max", "mean", "min"}}
    fam: dict[str, list[str]] = {
        "lag": [],
        "rolling": [],
        "forecast_now": [],
        "forecast_window": [],
        "static": [],
        "raw_observation": [],
    }
    for c in columns:
        if c in lag_names:
            fam["lag"].append(c)
        elif c in roll_names:
            fam["rolling"].append(c)
        elif c in fc_now:
            fam["forecast_now"].append(c)
        elif c in fc_win:
            fam["forecast_window"].append(c)
        elif c in static_cols:
            fam["static"].append(c)
        elif c in dyn_cols:
            fam["raw_observation"].append(c)
    return fam


# ---------------------------------------------------------------------------
# 构造结果的独立复核
# ---------------------------------------------------------------------------
def audit_built_features(
    dataset: Dataset,
    fs: FeatureSet,
    cfg: Config,
    *,
    max_grids: int = 3,
    max_lag_features: int = 12,
    max_rolling_features: int = 12,
) -> dict:
    """抽样复核：把构造出来的 lag / rolling / 预报窗口列拿去和原始观测逐点比对。

    为什么值得单独做：``features.py`` 的 bug 往往是"符号写反""shift 放在 rolling
    之后"这类，眼睛看不出来、模型也不会报错，只会让指标好看一点点。
    """
    from .leakage import check_time_lag, check_time_rolling

    df = dataset.df
    grid_col, ts_col = dataset.grid_id_col, dataset.timestamp_col
    step = float(cfg.get("task.time_step_hours", 1.0))
    min_lag = int(cfg.min_feature_lag)

    grids = list(pd.unique(df[grid_col]))[: max_grids]
    sample = df[df[grid_col].isin(grids)].copy()
    lag_feats = [c for c in fs.families.get("lag", [])][:max_lag_features]
    roll_feats = [c for c in fs.families.get("rolling", [])][:max_rolling_features]
    for c in lag_feats + roll_feats:
        sample[c] = fs.X.loc[sample.index, c]

    checked, failures = 0, []
    for feat in lag_feats:
        var, k = feat.rsplit("_lag", 1)
        try:
            check_time_lag(
                sample,
                grid_col=grid_col,
                time_col=ts_col,
                base_col=var,
                feat_col=feat,
                lag_steps=int(k),
                min_lag=min_lag,
                step_hours=step,
                name=feat,
            )
            checked += 1
        except Exception as exc:  # 只想收集，不想在这里中断流程
            failures.append({"feature": feat, "error": str(exc)})

    rolling_checked, rolling_failures = 0, []
    for feat in roll_feats:
        var, rest = feat.rsplit("_r", 1)
        n_digits = len(rest) - len(rest.lstrip("0123456789"))
        if n_digits == 0 or n_digits == len(rest):
            rolling_failures.append({"feature": feat, "error": f"无法解析滚动特征名 {feat!r}"})
            continue
        window, stat = int(rest[:n_digits]), rest[n_digits:]
        try:
            check_time_rolling(
                sample,
                grid_col=grid_col,
                time_col=ts_col,
                base_col=var,
                feat_col=feat,
                lag_steps=min_lag,
                window=window,
                stat=stat,
                min_lag=min_lag,
                step_hours=step,
                name=feat,
            )
            rolling_checked += 1
        except Exception as exc:
            rolling_failures.append({"feature": feat, "error": str(exc)})

    # 预报类：用独立算法（逐时刻切片取统计量）复核未来窗口
    fc_failures = []
    for feat in fs.families.get("forecast_window", []):
        var, rest = feat.split("_fw", 1)
        stat = next((s for s in ("mean", "min", "max") if rest.endswith(s)), None)
        if stat is None or not rest[: len(rest) - len(stat)].isdigit():
            raise FeatureError(f"无法解析预报窗口特征名 {feat!r}")
        width = int(rest[: len(rest) - len(stat)])
        for gid in grids:
            orig = df[df[grid_col] == gid].sort_values(ts_col)
            lookup = dict(zip(orig[ts_col], orig[var]))
            for ts, got in zip(orig[ts_col].head(6), fs.X.loc[orig.index[:6], feat]):
                window = [lookup.get(ts + pd.Timedelta(hours=step * j)) for j in range(width)]
                present = [w for w in window if w is not None and not pd.isna(w)]
                if len(present) < width:
                    # 窗口不完整（缺报或越界）：严格策略下特征应为 NaN。
                    # 有值反而说明某处做了填补，必须报出来。
                    if not pd.isna(got):
                        fc_failures.append(
                            {"feature": feat, "grid": gid, "t": str(ts), "why": "窗口不完整却有值"}
                        )
                        break
                    continue
                expect = {"max": max, "mean": np.mean, "min": min}[stat](present)
                if not np.isclose(got, expect, atol=1e-6):
                    fc_failures.append({"feature": feat, "grid": gid, "t": str(ts), "why": "数值不符"})
                    break

    return {
        "grids_sampled": [str(g) for g in grids],
        "lag_features_checked": checked,
        "lag_failures": failures,
        "rolling_features_checked": rolling_checked,
        "rolling_failures": rolling_failures,
        "forecast_failures": fc_failures,
        "passed": not failures and not rolling_failures and not fc_failures,
        "note": "抽样复核（lag + rolling + 预报窗口），不是全量；用于抓构造逻辑写错，不替代单元测试",
    }
