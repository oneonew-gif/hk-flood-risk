"""训练/验证/测试切分。

为什么这块比其他模块都重要
--------------------------
GIS 项目最常见的崩盘点不是模型，是切分：一旦按行随机切，相邻网格同时进了训练
和测试，模型等于"背答案"，AUC 会虚高十几个点，而且答辩现场无法自证。

本模块提供四种策略，全部返回**行位置索引**（不是布尔掩码以外的花样）：

- ``random``        仅用于程序冒烟测试，真实数据上会同时泄漏空间与时间
- ``temporal``      按时间切，同址未来时刻 —— 贴近业务运行场景
- ``spatial_block`` 按空间块整体切，未见地点 —— 防空间自相关泄漏
- ``spacetime``     未见空间块 + 未来时段 —— 最严格，建议作为正式报告的主结果

没有坐标或分块列时**不会偷偷退化成随机切分**，而是直接报错 —— 因为静默降级
正是这类项目最容易犯的错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, train_test_split

from .config import Config
from .io import Dataset


class SplitError(RuntimeError):
    """无法构造合法切分。"""


@dataclass
class SplitPlan:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def summary_with_labels(self, y: np.ndarray) -> dict[str, Any]:
        def _c(idx: np.ndarray) -> dict[str, Any]:
            if len(idx) == 0:
                return {"n": 0, "positives": 0, "positive_rate": None}
            lab = y[idx]
            return {
                "n": int(len(idx)),
                "positives": int(lab.sum()),
                "positive_rate": round(float(lab.mean()), 4),
            }

        return {
            "strategy": self.name,
            "train": _c(self.train_idx),
            "val": _c(self.val_idx),
            "test": _c(self.test_idx),
            **self.meta,
        }

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train_idx": self.train_idx.tolist(),
            "val_idx": self.val_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
        }


# ---------------------------------------------------------------------------
# 空间块
# ---------------------------------------------------------------------------
def assign_blocks(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """返回每行的空间块标签。

    优先用数据自带的行政区/分块列；没有则用坐标做 KMeans 聚类；
    两者都没有就报错，绝不退化成随机。
    """
    explicit = cfg.get("split.spatial_block_column", None)
    if explicit and explicit in df.columns:
        return df[explicit].astype(str).to_numpy()
    if "spatial_block_id" in df.columns:
        return df["spatial_block_id"].astype(str).to_numpy()

    coord_pairs = (("x_km", "y_km"), ("lon", "lat"))
    coords = None
    used = None
    for pair in coord_pairs:
        if all(c in df.columns for c in pair):
            coords = df[list(pair)].to_numpy(dtype=float)
            used = pair
            break
    if coords is None:
        raise SplitError(
            "数据既没有 spatial_block_column / spatial_block_id，也没有 (x_km,y_km) 或 "
            "(lon,lat) 坐标，无法构造空间切分。请让数据侧补上网格质心坐标或行政区编码；"
            "在补齐之前，本框架不会退化成随机切分"
        )

    k = int(cfg.get("split.spatial_blocks", 8))
    seed = int(cfg.get("split.seed", 42))
    # 只在唯一网格上聚类，避免长时间序列样本量放大权重
    uniq = df.drop_duplicates(subset=["grid_id"])[list(used)].to_numpy(dtype=float)
    km = KMeans(n_clusters=min(k, len(uniq)), n_init=10, random_state=seed).fit(uniq)
    uniq_ids = df.drop_duplicates(subset=["grid_id"])["grid_id"].to_numpy()
    mapping = {gid: f"K{lab}" for gid, lab in zip(uniq_ids, km.labels_)}
    return df["grid_id"].map(mapping).to_numpy()


def _purge_delta(cfg: Config) -> pd.Timedelta:
    """标签窗口的重叠清洗带宽度（purge）。

    标签是 [t, t+h) 的窗口标签。若训练集末行是 t=T-1、验证集从 T 开始，
    训练行的标签窗口就伸进了验证段：两段共享同一批"水浸事件"，评估偏乐观。
    标准做法是在两段之间留出 h 的清洗带。static 模式没有时间维度，恒为 0。
    """
    if cfg.mode != "dynamic" or not bool(cfg.get("split.purge_horizon", True)):
        return pd.Timedelta(0)
    step = float(cfg.get("task.time_step_hours", 1.0))
    return pd.Timedelta(hours=cfg.horizon_steps * step)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def make_split_plan(dataset: Dataset, cfg: Config) -> SplitPlan:
    df = dataset.df
    y = dataset.y
    strategy = cfg.effective_split_strategy()
    test_size = float(cfg.get("split.test_size"))
    val_size = float(cfg.get("split.val_size"))
    seed = int(cfg.get("split.seed", 42))
    purge = _purge_delta(cfg)
    n = len(df)
    if n < 30:
        raise SplitError(f"样本量过小（{n} 行），无法切出三份数据集")

    if strategy == "random":
        idx = np.arange(n)
        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=_safe_stratify(y)
        )
        rel_val = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=rel_val, random_state=seed, stratify=_safe_stratify(y[train_val_idx])
        )
        plan = SplitPlan("random", np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))

    elif strategy == "temporal":
        train_idx, val_idx, test_idx, meta = _temporal_masks(df, test_size, val_size, purge)
        plan = SplitPlan("temporal", train_idx, val_idx, test_idx, meta)

    elif strategy == "spatial_block":
        blocks = assign_blocks(df, cfg)
        train_idx, val_idx, test_idx, meta = _block_masks(blocks, test_size, val_size, seed)
        plan = SplitPlan("spatial_block", train_idx, val_idx, test_idx, meta)

    elif strategy == "spacetime":
        if "timestamp" not in df.columns:
            raise SplitError("spacetime 策略需要 timestamp 列")
        blocks = assign_blocks(df, cfg)
        plan = _spacetime_plan(df, blocks, test_size, val_size, seed, purge)

    else:  # pragma: no cover - config 已校验
        raise SplitError(f"未知切分策略：{strategy}")

    _assert_usable(plan, y)

    # 按时间切分时，同一批地点必然跨集合出现 —— 这是该策略的定义，不是 bug。
    # 但必须如实记录共享了多少个空间块，方便在报告里交代这一层乐观偏差。
    if strategy == "temporal":
        try:
            blocks = assign_blocks(df, cfg)
            plan.meta["blocks_shared_with_test"] = int(
                len(set(np.unique(blocks[plan.train_idx])) & set(np.unique(blocks[plan.test_idx])))
            )
            plan.meta["blocks_total"] = int(len(np.unique(blocks)))
        except SplitError:
            pass
    return plan


def cv_splits(dataset: Dataset, cfg: Config, pool_idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """构造交叉验证折，**必须与切分策略同源**，否则防泄漏三道闸会被 CV 绕过。"""
    if len(pool_idx) < 20:
        raise SplitError(f"CV 可用样本仅 {len(pool_idx)} 行，太少")
    folds = int(cfg.get("split.cv_folds"))
    df = dataset.df
    strategy = cfg.effective_split_strategy()

    if strategy in {"temporal", "spacetime"}:
        ts = pd.to_datetime(df.loc[pool_idx, "timestamp"]).to_numpy()
        uniq = np.sort(np.unique(ts))
        if len(uniq) < folds + 1:
            raise SplitError(f"时间点仅 {len(uniq)} 个，不足以做 {folds} 折时序 CV")
        purge = _purge_delta(cfg)
        edges = np.linspace(0, len(uniq), folds + 2).astype(int)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, folds + 1):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo:
                continue
            va_start = uniq[lo]
            va_end = uniq[hi] if hi < len(uniq) else None
            # 折间同样要 purge：否则训练折末尾的标签窗口会伸进验证折
            tr_cut = np.datetime64(pd.Timestamp(va_start) - purge)
            tr_mask = ts < tr_cut
            va_mask = ts >= va_start if va_end is None else (ts >= va_start) & (ts < va_end)
            if tr_mask.sum() < 10 or va_mask.sum() < 10:
                continue
            out.append((pool_idx[tr_mask], pool_idx[va_mask]))
        if not out:
            raise SplitError("时序 CV 未能生成任何折，请减少 cv_folds 或检查时间跨度")
        return out

    # 空间类策略：按块分组
    blocks_all = assign_blocks(df, cfg)
    blocks = blocks_all[pool_idx]
    n_groups = len(np.unique(blocks))
    if n_groups < 2:
        raise SplitError(
            f"训练池内只有 {n_groups} 个空间块，无法做分组 CV。"
            f"请增加 spatial_blocks 或放宽 test/val 比例"
        )
    # GroupKFold 要求 n_splits <= 组数，块少时自动降折数并记录
    n_splits = int(min(folds, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    out = []
    for tr_rel, va_rel in gkf.split(pool_idx, groups=blocks):
        out.append((pool_idx[tr_rel], pool_idx[va_rel]))
    return out


# ---------------------------------------------------------------------------
# 各策略实现
# ---------------------------------------------------------------------------
def _temporal_masks(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    purge: pd.Timedelta = pd.Timedelta(0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"])
    uniq = np.sort(ts.unique())
    n_t = len(uniq)
    if n_t < 5:
        raise SplitError(f"时间点仅 {n_t} 个，无法按时间切出 train/val/test")
    test_start_i = _clamp(int(round(n_t * (1.0 - test_size))), 1, n_t - 1)
    val_start_i = _clamp(int(round(n_t * (1.0 - test_size - val_size))), 1, test_start_i - 1)
    test_start, val_start = uniq[test_start_i], uniq[val_start_i]

    # purge：把上一段末尾"标签窗口会伸进下一段"的行剥掉。
    # 标签是 [t, t+h) 的窗口，训练集最后一行若贴着 val_start，它的标签窗口
    # 与验证段首行共享同一批事件 —— 两个集合看到的是同一场雨，评估偏乐观。
    train_cut = np.datetime64(pd.Timestamp(val_start) - purge)
    val_cut = np.datetime64(pd.Timestamp(test_start) - purge)

    v = ts.to_numpy()
    train_idx = np.flatnonzero(v < train_cut)
    val_idx = np.flatnonzero((v >= val_start) & (v < val_cut))
    test_idx = np.flatnonzero(v >= test_start)

    def _edge(idx: np.ndarray, which: str) -> str | None:
        if len(idx) == 0:
            return None
        return str(pd.Timestamp(v[idx].min() if which == "min" else v[idx].max()))

    purged = int((~np.isin(np.arange(len(df)), np.concatenate([train_idx, val_idx, test_idx]))).sum())
    meta = {
        # 边界一律写**实际**首末时点，不要写成对方的起点 —— 那会让人误读成有重叠
        "time_boundaries": {
            "train_start": _edge(train_idx, "min"),
            "train_end": _edge(train_idx, "max"),
            "val_start": _edge(val_idx, "min"),
            "val_end": _edge(val_idx, "max"),
            "test_start": _edge(test_idx, "min"),
            "test_end": _edge(test_idx, "max"),
        },
        "purge_hours": float(purge.total_seconds() / 3600.0),
        "purged_rows": purged,
        "note": "同一批地点跨时间切分：贴近业务运行（用过去预测未来），"
        "但不检验对未见地点的泛化能力。purged_rows 是为消除标签窗口重叠"
        "而剥离的行，属预期行为",
    }
    return train_idx, val_idx, test_idx, meta


def _block_masks(
    blocks: np.ndarray, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    uniq = np.unique(blocks)
    if len(uniq) < 3:
        raise SplitError(f"空间块只有 {len(uniq)} 个，无法切出 train/val/test 三份")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_size)))
    n_val = max(1, int(round(len(uniq) * val_size)))
    if len(uniq) - n_test - n_val < 1:
        n_test, n_val = 1, 1
    test_blocks = set(uniq[order[:n_test]])
    val_blocks = set(uniq[order[n_test : n_test + n_val]])
    train_blocks = set(uniq[order[n_test + n_val :]])

    in_block = lambda s: np.isin(blocks, list(s))  # noqa: E731
    train_idx = np.flatnonzero(in_block(train_blocks))
    val_idx = np.flatnonzero(in_block(val_blocks))
    test_idx = np.flatnonzero(in_block(test_blocks))
    meta = {
        "blocks": {
            "train": sorted(train_blocks),
            "val": sorted(val_blocks),
            "test": sorted(test_blocks),
        },
        "note": "整块进出：同一空间块不会同时出现在两个集合里，用于检验对未见地点的泛化",
    }
    return train_idx, val_idx, test_idx, meta


def _spacetime_plan(
    df: pd.DataFrame,
    blocks: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    purge: pd.Timedelta = pd.Timedelta(0),
) -> SplitPlan:
    tr, va, te, block_meta = _block_masks(blocks, test_size, val_size, seed)
    ts = pd.to_datetime(df["timestamp"])
    uniq = np.sort(ts.unique())
    n_t = len(uniq)
    if n_t < 5:
        raise SplitError(f"时间点仅 {n_t} 个，无法做时空切分")
    test_start_i = _clamp(int(round(n_t * (1.0 - test_size))), 1, n_t - 1)
    test_start = uniq[test_start_i]
    cut = np.datetime64(pd.Timestamp(test_start) - purge)
    v = ts.to_numpy()

    # 测试集：未见空间块 + 未来时段；验证集：另一批未见块 + 较早时段
    test_idx = np.intersect1d(te, np.flatnonzero(v >= test_start))
    val_idx = np.intersect1d(va, np.flatnonzero(v < cut))
    train_idx = np.intersect1d(tr, np.flatnonzero(v < cut))

    used = np.zeros(len(df), dtype=bool)
    used[np.concatenate([train_idx, val_idx, test_idx])] = True
    meta = dict(block_meta)
    meta["time_boundaries"] = {"test_start": str(pd.Timestamp(test_start))}
    meta["purge_hours"] = float(purge.total_seconds() / 3600.0)
    meta["unused_rows"] = int((~used).sum())
    meta["note"] = (
        "最严格：测试集既未见地点也未见时段；未使用的行（测试块的历史时段"
        "与清洗带内的行）不参与训练也不参与评估，属预期行为"
    )
    return SplitPlan("spacetime", train_idx, val_idx, test_idx, meta)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _safe_stratify(y: np.ndarray):
    counts = np.bincount(y.astype(int))
    counts = counts[counts > 0]
    return y if len(counts) > 1 and counts.min() >= 2 else None


def _assert_usable(plan: SplitPlan, y: np.ndarray) -> None:
    sets = {
        "train": set(plan.train_idx.tolist()),
        "val": set(plan.val_idx.tolist()),
        "test": set(plan.test_idx.tolist()),
    }
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise SplitError(f"{a} 与 {b} 存在 {len(sets[a] & sets[b])} 行重叠，切分非法")
    for name, idx in (("train", plan.train_idx), ("val", plan.val_idx), ("test", plan.test_idx)):
        if len(idx) == 0:
            raise SplitError(f"{name} 为空，请调整 test_size / val_size 或时间跨度")
        labels = y[idx]
        if labels.sum() == 0 or labels.sum() == len(labels):
            raise SplitError(
                f"{name} 只有单一类别（正例 {int(labels.sum())}/{len(labels)}），"
                f"无法评估；检查切分比例或标签的时空分布"
            )
