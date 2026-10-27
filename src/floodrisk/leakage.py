"""泄漏闸门（三道）。

1. **特征黑名单** —— 名字命中正则即中止。典型泄漏字段：水浸深度、水浸报告数、
   事后清淤/巡查记录、应急响应次数。这些东西在预测时点根本不可知。
2. **时序对齐** —— lag/rolling 特征的位移不得小于锚定口径推出的最小合法滞后
   （``Config.min_feature_lag``），并做**按时间轴**的数值校验：
   ``feature[t] == base[t - lag*step]``。按行 shift 在面板有缺口时会错位，
   这是数值校验存在的原因。
3. **切分不相交** —— 行级与空间块级双重检查。

另附**单变量 AUC 扫描**：某个特征单独就能把标签分得几乎干净（AUC>0.97），
通常不是模型太强，而是信息泄漏。它只告警不中止 —— 小样本或极强代理变量也可能
出现这种数值，需要你人工判断。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import Config


class LeakageError(RuntimeError):
    """检测到信息泄漏，拒绝继续训练。"""


# ---------------------------------------------------------------------------
# 闸门 1：特征黑名单
# ---------------------------------------------------------------------------
def check_feature_names(features: Iterable[str], cfg: Config) -> list[str]:
    """命中黑名单即抛错；返回命中的 (feature, pattern) 说明供报告使用。"""
    patterns = [re.compile(p) for p in (cfg.get("leakage.forbidden_patterns", []) or [])]
    exact = {str(x) for x in (cfg.get("leakage.forbidden_exact", []) or [])}
    target = cfg.get("task.target", "flood_label")

    hits: list[tuple[str, str]] = []
    for feat in features:
        name = str(feat)
        if name == target:
            hits.append((name, "与标签列同名"))
            continue
        if name in exact:
            hits.append((name, "命中 leakage.forbidden_exact"))
            continue
        for pat in patterns:
            if pat.search(name):
                hits.append((name, f"命中黑名单正则 {pat.pattern!r}"))
                break

    if hits:
        detail = "\n".join(f"  - {n}: {r}" for n, r in hits)
        raise LeakageError(
            "以下特征疑似标签期信息（预测时点不可知），已拒绝训练：\n"
            f"{detail}\n"
            "处理方式：从 features.exclude 中剔除，或在构造数据时不要生成这类字段。"
            "若确信其合法（例如排水'设计标准'而非'事后维护记录'），"
            "请把它加入 leakage.forbidden_exact 的例外说明并在此说明理由"
        )
    return []


# ---------------------------------------------------------------------------
# 闸门 2：时序对齐（按时间位移，不是按行位移）
# ---------------------------------------------------------------------------
def _aligned_pair(
    df: pd.DataFrame,
    grid_col: str,
    time_col: str,
    base_col: str,
    feat_col: str,
    step_hours: float,
):
    """逐网格把 base / feature 重索引到完整时间轴上。

    为什么必须重索引：面板数据一旦有缺口（某网格某小时缺报），``df.shift(3)``
    是按**行**位移，实际对齐到的可能是 4 小时前。按时间轴重索引后，
    ``shift(k)`` 才真正等于"k 步之前的那条观测"。
    """
    step = pd.Timedelta(hours=float(step_hours))
    for gid, sub in df.groupby(grid_col, sort=False):
        sub = sub.dropna(subset=[time_col]).set_index(time_col).sort_index()
        if sub.index.duplicated().any():
            raise LeakageError(
                f"网格 {gid} 的时间索引有重复，先去重再做时序校验"
            )
        full = pd.date_range(sub.index.min(), sub.index.max(), freq=step)
        yield gid, sub[base_col].reindex(full), sub[feat_col].reindex(full)


def check_time_lag(
    df: pd.DataFrame,
    *,
    grid_col: str,
    time_col: str,
    base_col: str,
    feat_col: str,
    lag_steps: int,
    min_lag: int,
    step_hours: float,
    name: str = "<unnamed>",
    atol: float = 1e-8,
) -> None:
    """数值级验证滞后特征：``feature[t] == base[t - lag*step]``（按时间，非按行）。

    同时检查 ``lag_steps >= min_lag``（min_lag 由锚定口径与数据约定推出，
    见 Config.min_feature_lag —— 它与 horizon 无关）。
    """
    if lag_steps < min_lag:
        raise LeakageError(
            f"特征 {name} 的滞后为 {lag_steps} 步，小于最小合法滞后 {min_lag} 步。"
            f"该值由 task.anchor 与 data.time_conventions 推出；"
            f"若数据约定已确认，可通过 min_lag_policy: derived 放开"
        )
    checked = 0
    for gid, base_full, feat_full in _aligned_pair(
        df, grid_col, time_col, base_col, feat_col, step_hours
    ):
        expected = base_full.shift(lag_steps)
        # (a) 期望值可得 → 必须相等
        mask = expected.notna() & feat_full.notna()
        checked += int(mask.sum())
        if mask.any() and not np.allclose(
            expected[mask].to_numpy(), feat_full[mask].to_numpy(), atol=atol
        ):
            bad = int(
                (~np.isclose(expected[mask].to_numpy(), feat_full[mask].to_numpy(), atol=atol)).sum()
            )
            raise LeakageError(
                f"特征 {name} 在网格 {gid} 有 {bad} 处与按时间位移 {lag_steps} 步的基准值不符。"
                f"最常见原因：用 df.shift() 按行位移，而面板存在时间缺口"
            )
        # (b) 期望值不可得（该时点在时间轴上根本没有观测）但特征有值
        #     → 只能是按行位移跨过了缺口，或用 ffill 之类把不存在的观测补了出来。
        #     这一条是 (a) 抓不到的：按行位移在缺口后的第一段恰好落在 NaN 上，
        #     (a) 的掩码会把它跳过，看起来"没问题"。
        orphan = expected.isna() & feat_full.notna()
        if orphan.any():
            raise LeakageError(
                f"特征 {name} 在网格 {gid} 有 {int(orphan.sum())} 处取值，"
                f"但按时间轴回看 {lag_steps} 步该时点并无观测（序列存在缺报）。"
                f"按行位移会把缺口前更早的观测当成 {lag_steps} 步前的值，"
                f"实际滞后与声明不符 —— 请先把每个网格重索引到完整时间轴再 shift"
            )
    if checked == 0:
        raise LeakageError(
            f"特征 {name} 没有任何可与基准列按时间对齐的样本，无法验证位移是否正确。"
            f"请检查时间列与步长设置"
        )


def check_time_rolling(
    df: pd.DataFrame,
    *,
    grid_col: str,
    time_col: str,
    base_col: str,
    feat_col: str,
    lag_steps: int,
    window: int,
    stat: str,
    min_lag: int,
    step_hours: float,
    name: str = "<unnamed>",
    atol: float = 1e-6,
) -> None:
    """数值级验证滚动统计只用了位移之后的过去窗口（按完整时间轴计算）。

    期望值 = ``base.shift(lag).rolling(window).<stat>()``（时间轴版）。
    """
    if lag_steps < min_lag:
        raise LeakageError(
            f"特征 {name} 的位移为 {lag_steps} 步，小于最小合法滞后 {min_lag} 步"
        )
    if window < 2:
        raise LeakageError(f"特征 {name} 的滚动窗口须 >= 2，当前 {window}")
    checked = 0
    for gid, base_full, feat_full in _aligned_pair(
        df, grid_col, time_col, base_col, feat_col, step_hours
    ):
        shifted = base_full.shift(lag_steps)
        strict = getattr(shifted.rolling(window), stat)()          # 要求窗口内全有观测
        loose = getattr(shifted.rolling(window, min_periods=1), stat)()  # 容忍缺测
        # (a) 窗口完整 → 必须严格相等
        mask = strict.notna() & feat_full.notna()
        checked += int(mask.sum())
        if mask.any() and not np.allclose(
            strict[mask].to_numpy(), feat_full[mask].to_numpy(), atol=atol
        ):
            bad = int(
                (~np.isclose(strict[mask].to_numpy(), feat_full[mask].to_numpy(), atol=atol)).sum()
            )
            raise LeakageError(
                f"特征 {name} 在网格 {gid} 有 {bad} 处与 "
                f"shift({lag_steps}).rolling({window}).{stat}() 不符，"
                f"滚动窗口可能包含了未来时点或位移按行而非按时间"
            )
        # (b) 窗口内有缺测（严格值不可得）但特征有值 → 必须等于"容忍缺测"的时间轴结果。
        #     若不等，说明窗口是按行跨缺口拼出来的：它把缺口前更早的观测算进了窗口，
        #     窗口的真实覆盖时段比声明的更长，也更容易碰到未来时点。
        partial = strict.isna() & feat_full.notna()
        if partial.any():
            bad = int(
                (~np.isclose(
                    loose[partial].to_numpy(), feat_full[partial].to_numpy(), atol=atol
                )).sum()
            )
            if bad:
                raise LeakageError(
                    f"特征 {name} 在网格 {gid} 有 {bad} 处与按时间轴计算的窗口值不符"
                    f"（该处窗口内有缺测）。按行位移会跨过缺口取到更早的观测，"
                    f"实际覆盖时段与声明不符；若确实要容忍缺测，"
                    f"应在完整时间轴上用 rolling(min_periods=1) 构造"
                )
    if checked == 0:
        raise LeakageError(
            f"特征 {name} 没有任何可与基准列按时间对齐的样本，无法验证窗口是否合法"
        )


# ---------------------------------------------------------------------------
# 闸门 3：切分不相交
# ---------------------------------------------------------------------------
def assert_split_disjoint(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    blocks: np.ndarray | None = None,
    *,
    strict_blocks: bool = True,
) -> None:
    pairs = {
        "train/val": (train_idx, val_idx),
        "train/test": (train_idx, test_idx),
        "val/test": (val_idx, test_idx),
    }
    for label, (a, b) in pairs.items():
        overlap = set(a.tolist()) & set(b.tolist())
        if overlap:
            raise LeakageError(f"切分重叠：{label} 共享 {len(overlap)} 行")

    if blocks is not None and strict_blocks:
        for label, (a, b) in pairs.items():
            shared = set(np.unique(blocks[a]).tolist()) & set(np.unique(blocks[b]).tolist())
            if shared:
                raise LeakageError(
                    f"空间块泄漏：{label} 共享空间块 {sorted(shared)[:8]}。"
                    f"随机切分会让相邻网格同时进入训练与评估，指标会虚高"
                )


# ---------------------------------------------------------------------------
# 单变量 AUC 扫描
# ---------------------------------------------------------------------------
def univariate_auc_scan(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    warn_threshold: float = 0.97,
) -> list[dict[str, Any]]:
    """逐个特征算单变量 AUC（取方向无关的最大值），标记可疑项。"""
    results: list[dict[str, Any]] = []
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return results
    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        if series.notna().sum() < 10:
            continue
        filled = series.fillna(series.median())
        if filled.nunique() <= 1:
            continue
        try:
            auc = float(roc_auc_score(y, filled.to_numpy()))
        except ValueError:  # pragma: no cover
            continue
        directional = max(auc, 1.0 - auc)
        results.append(
            {
                "feature": col,
                "univariate_auc": round(directional, 4),
                "direction": "positive" if auc >= 0.5 else "negative",
                "suspect_leakage": directional >= warn_threshold,
            }
        )
    results.sort(key=lambda r: r["univariate_auc"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 汇总审计
# ---------------------------------------------------------------------------
def audit_features(dataset, cfg: Config, *, scan_idx: np.ndarray) -> dict[str, Any]:
    """一次性跑完三道闸中可静态检查的部分，产出可写入 manifest 的审计报告。

    ``scan_idx``：单变量 AUC 扫描允许使用的行（应传训练段行索引）。
    AUC 扫描消耗标签，若用全量数据，验证/测试标签会经审计报告回流到
    特征取舍决策 —— 这也是泄漏。因此扫描被强制限制在调用方给定的行内；
    黑名单检查不使用标签，始终覆盖全部特征名。
    """
    features = list(dataset.features)
    check_feature_names(features, cfg)

    warn_thr = float(cfg.get("leakage.warn_univariate_auc", 0.97))
    scan = univariate_auc_scan(
        dataset.X.iloc[scan_idx], dataset.y[scan_idx], warn_threshold=warn_thr
    )
    suspects = [r for r in scan if r["suspect_leakage"]]

    return {
        "features_checked": len(features),
        "blacklist_hits": [],
        "univariate_auc": scan,
        "suspect_features": [r["feature"] for r in suspects],
        "warn_threshold": warn_thr,
        "auc_scan_scope": f"仅传入的 {int(len(scan_idx))} 行（应为训练段），验证/测试标签不参与",
        "note": (
            "单变量 AUC 偏高不等于一定泄漏，但需要人工确认该变量的可观测时点；"
            "若它只能在事件发生后获取，必须剔除"
        ),
    }
