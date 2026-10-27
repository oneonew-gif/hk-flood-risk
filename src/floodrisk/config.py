"""配置加载与校验。

设计原则
--------
1. ``configs/base.yaml`` 是所有默认值的唯一来源，代码里不写魔法默认值。
2. 非法配置**当场抛错**而不是悄悄兜底 —— 尤其三级阈值，缺 ``cutoffs`` 时必须
   中止，避免程序替使用者做了一个它没资格做的决定。
3. 每次运行把配置哈希写进 manifest，保证结果可追溯。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置非法，直接中止运行。"""


_MISSING = object()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


class Config:
    """极简的不可变式配置容器：点号取值 + 严格校验。"""

    def __init__(self, data: dict, source: str | Path | None = None):
        if not isinstance(data, dict):
            raise ConfigError("配置根节点必须是映射（dict）")
        self._data = data
        self.source = str(source) if source is not None else "<memory>"
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ 构造
    @classmethod
    def load(
        cls,
        path: str | Path,
        overrides: dict | None = None,
        *,
        _chain: tuple[str, ...] = (),
    ) -> "Config":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"配置文件不存在：{p}")
        if str(p.resolve()) in _chain:
            raise ConfigError(f"extends 出现循环引用：{p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件根节点必须是映射：{p}")

        # 支持 extends: base.yaml（相对当前文件解析），子文件只需写差异项
        parent = raw.pop("extends", None)
        if parent:
            base_path = (p.parent / str(parent)).resolve()
            base_cfg = cls.load(base_path, None, _chain=_chain + (str(p.resolve()),))
            raw = _deep_merge(base_cfg.to_dict(), raw)

        if overrides:
            raw = _deep_merge(raw, overrides)
        cfg = cls(raw, p)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, data: dict, overrides: dict | None = None) -> "Config":
        if overrides:
            data = _deep_merge(data, overrides)
        cfg = cls(data)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------ 读取
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"缺少配置项：{dotted}")
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise ConfigError(f"配置项不能为空：{dotted}")
        return value

    def section(self, name: str) -> dict:
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"配置段必须是映射：{name}")
        return value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data, default=str))

    def fingerprint(self) -> str:
        blob = json.dumps(self._data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ 派生
    @property
    def mode(self) -> str:
        return self.get("task.mode")

    @property
    def is_dynamic(self) -> bool:
        return self.mode == "dynamic"

    @property
    def horizon_steps(self) -> int:
        """标签窗口长度（步）。只作用于标签侧，不约束特征滞后。"""
        step = float(self.get("task.time_step_hours", 1.0))
        horizon = float(self.get("task.horizon_hours", 0.0))
        if step <= 0:
            raise ConfigError("task.time_step_hours 必须 > 0")
        return max(1, int(math.ceil(horizon / step)))

    @property
    def anchor(self) -> str:
        """行动时间锚：issue_time（业务口径）或 window_end（复盘口径）。"""
        return self.get("task.anchor", "issue_time")

    def _var_min_lag(self, spec: dict) -> int:
        """单个变量在当前锚定口径下的最小合法滞后（步）。

        - interval + anchor=end  ：值在时间戳时刻已聚合完成 → 延迟 = reporting_delay
        - interval + anchor=start：值在时间戳 + 1 步才完成   → 延迟 = 1 + delay
        - instantaneous          ：瞬时读数，时间戳时刻即得   → 延迟 = delay
        - window_end 锚定下，以上全部再叠加 horizon_steps
          （特征必须整段早于标签窗口打开）
        """
        agg = spec.get("aggregate", "interval")
        delay = int(spec.get("reporting_delay_steps", 0))
        if agg == "static":
            return 0
        extra = 1 if (agg == "interval" and spec.get("anchor") == "start") else 0
        if self.anchor == "window_end":
            return self.horizon_steps + delay + extra
        return delay + extra

    @property
    def min_feature_lag(self) -> int:
        """全局特征滞后下限：所有约定变量中最严格者。

        issue_time 口径下受 min_lag_policy 控制：
        - conservative（默认）：取 1 —— A 书面确认时间戳约定前的保守值，
          代价是放弃最近一小时雨量，收益是杜绝"约定读反"这类整批泄漏
        - derived：按 time_conventions 逐变量推出
        window_end 口径下 horizon 强制参与，不受 policy 豁免。
        """
        if self.anchor == "window_end":
            derived = max(
                (self._var_min_lag(s) for s in (self.get("data.time_conventions", {}) or {}).values()),
                default=self.horizon_steps,
            )
            return max(derived, self.horizon_steps)
        policy = self.get("features.temporal.min_lag_policy", "conservative")
        if policy == "conservative":
            return 1
        if policy == "derived":
            conv = self.get("data.time_conventions", {}) or {}
            return max((self._var_min_lag(s) for s in conv.values()), default=0)
        raise ConfigError(
            f"features.temporal.min_lag_policy 只支持 conservative / derived，当前 {policy!r}"
        )

    def effective_split_strategy(self) -> str:
        strategy = self.get("split.strategy", "auto")
        if strategy != "auto":
            return strategy
        return "temporal" if self.is_dynamic else "spatial_block"

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        w: list[str] = []

        # ---- task
        mode = self.get("task.mode", None)
        if mode not in {"dynamic", "static"}:
            raise ConfigError(f"task.mode 只能是 dynamic / static，当前为 {mode!r}")
        if not self.get("task.target", None):
            raise ConfigError("task.target 不能为空")

        # ---- data
        if not self.get("data.input_path", None):
            raise ConfigError("data.input_path 不能为空")
        fmt = self.get("data.format", "csv")
        if fmt not in {"csv", "parquet", "xlsx"}:
            raise ConfigError(f"data.format 不支持：{fmt!r}")

        # ---- task.anchor
        anchor = self.get("task.anchor", "issue_time")
        if anchor not in {"issue_time", "window_end"}:
            raise ConfigError(
                f"task.anchor 只能是 issue_time / window_end，当前 {anchor!r}。"
                f"issue_time=行时间戳是预测发起时刻（业务口径）；"
                f"window_end=行时间戳是标签窗口末端（复盘口径）"
            )
        if mode == "static" and self.get("task.anchor", None) == "window_end":
            raise ConfigError("static 模式没有时间维度，task.anchor 无意义，请删除或改为 issue_time")

        # ---- data.time_conventions
        conventions = self.get("data.time_conventions", {}) or {}
        if not isinstance(conventions, dict):
            raise ConfigError("data.time_conventions 必须是 映射（变量 -> 约定）")
        for var, spec in conventions.items():
            if not isinstance(spec, dict):
                raise ConfigError(f"time_conventions.{var} 必须是映射")
            agg = spec.get("aggregate", None)
            if agg not in {"interval", "instantaneous", "static"}:
                raise ConfigError(
                    f"time_conventions.{var}.aggregate 只支持 interval / instantaneous / static，"
                    f"当前 {agg!r}。不知道约定就别猜 —— lag 数值的安全性完全取决于它"
                )
            if agg == "interval" and spec.get("anchor") not in {"start", "end"}:
                raise ConfigError(
                    f"time_conventions.{var} 是 interval 聚合，必须声明 anchor: start|end"
                    f"（时间戳标记的是时段起点还是终点）"
                )
            if int(spec.get("reporting_delay_steps", 0)) < 0:
                raise ConfigError(f"time_conventions.{var}.reporting_delay_steps 不能为负")

        # ---- features
        lags = self.get("features.temporal.lags", []) or []
        if mode == "static" and lags:
            raise ConfigError(
                "static 模式下不允许配置 features.temporal.lags —— 静态易发性模型"
                "使用时间滞后特征属于时序泄漏，请把 lags 置空或改用 dynamic 模式"
            )
        if mode == "dynamic":
            min_lag = self.min_feature_lag  # 会顺带校验 min_lag_policy 合法性
            bad = [x for x in lags if int(x) < min_lag]
            if bad:
                raise ConfigError(
                    f"滞后步数 {bad} 小于最小合法滞后 {min_lag} 步"
                    f"（anchor={anchor}，min_lag_policy={self.get('features.temporal.min_lag_policy', 'conservative')}）。"
                    f"issue_time 口径下该值由 data.time_conventions 的聚合约定与上报延迟推出，"
                    f"与 horizon 无关；确认数据约定后可将 min_lag_policy 改为 derived 以放开更近的滞后"
                )
        for win in self.get("features.temporal.rolling_windows", []) or []:
            if int(win) < 2:
                raise ConfigError("rolling_windows 中的窗口须 >= 2")

        # ---- features.forecast（预报类：发起时刻已知的未来值）
        forecast = self.get("features.forecast", {}) or {}
        if not isinstance(forecast, dict):
            raise ConfigError("features.forecast 必须是映射")
        if forecast.get("enabled", False):
            stats = forecast.get("window_stats", ["max"])
            if not isinstance(stats, list) or not set(stats) <= {"max", "mean", "min"}:
                raise ConfigError(
                    f"features.forecast.window_stats 只支持 max / mean / min，当前 {stats!r}"
                )
            if not forecast.get("variables", []):
                raise ConfigError("features.forecast.enabled=true 时必须列出 variables")

        # ---- split
        strategy = self.get("split.strategy", "auto")
        if strategy not in {"auto", "random", "temporal", "spatial_block", "spacetime"}:
            raise ConfigError(f"split.strategy 不支持：{strategy!r}")
        if strategy == "temporal" and mode == "static":
            raise ConfigError("static 模式不能按时间切分（每个网格只有一行），请用 spatial_block")
        if strategy == "spatial_block" and mode == "dynamic":
            w.append(
                "dynamic 模式选择 spatial_block：会保留同一批地点跨时间的数据，"
                "若要同时防空间与时间泄漏请改用 spacetime"
            )
        if strategy == "random" and self.get("project.provenance_override", None) != "SYNTHETIC":
            w.append(
                "split.strategy=random 只应用于程序冒烟测试；真实数据上会同时产生"
                "空间与时间泄漏，指标不可引用"
            )
        for key in ("split.test_size", "split.val_size"):
            val = float(self.get(key))
            if not 0 < val < 1:
                raise ConfigError(f"{key} 必须在 (0, 1) 区间内，当前 {val}")
        if float(self.get("split.test_size")) + float(self.get("split.val_size")) >= 1:
            raise ConfigError("test_size + val_size 必须 < 1")
        if int(self.get("split.spatial_blocks", 1)) < 3 and self.effective_split_strategy() in {
            "spatial_block",
            "spacetime",
        }:
            raise ConfigError("spatial_blocks 至少为 3，才能切出 train/val/test 三份")
        if int(self.get("split.cv_folds", 0)) < 2:
            raise ConfigError("split.cv_folds 至少为 2")

        # ---- train
        imbalance = self.get("train.imbalance.strategy", "none")
        if imbalance not in {"none", "scale_pos_weight", "max_delta_step"}:
            raise ConfigError(f"train.imbalance.strategy 不支持：{imbalance!r}")
        if self.get("train.eval_metric", "aucpr") not in {"aucpr", "auc", "logloss", "error"}:
            raise ConfigError("train.eval_metric 只支持 aucpr / auc / logloss / error")

        # ---- tuning
        backend = self.get("tuning.backend", "random")
        if backend not in {"random", "grid", "optuna"}:
            raise ConfigError(f"tuning.backend 不支持：{backend!r}")
        if self.get("tuning.enabled", False) and int(self.get("tuning.n_iter", 0)) < 1:
            raise ConfigError("tuning.enabled=true 时 tuning.n_iter 必须 >= 1")
        if self.get("tuning.metric", "average_precision") not in {
            "average_precision",
            "roc_auc",
            "f1",
            "recall",
        }:
            raise ConfigError("tuning.metric 只支持 average_precision / roc_auc / f1 / recall")

        # ---- calibration
        method = self.get("calibration.method", "none")
        if method not in {"none", "platt", "isotonic"}:
            raise ConfigError(f"calibration.method 不支持：{method!r}")
        if method != "none" and float(self.get("split.val_size")) <= 0:
            raise ConfigError("启用概率校准时必须有独立验证集（split.val_size > 0）")

        # ---- tiering（重点：绝不提供 0.33/0.66 之类的默认阈值）
        t_strategy = self.get("tiering.strategy", None)
        if t_strategy not in {"fixed", "quantile", "optimize_recall", "cost"}:
            raise ConfigError(
                f"tiering.strategy 只支持 fixed / quantile / optimize_recall / cost，当前 {t_strategy!r}"
            )
        if t_strategy == "fixed":
            cutoffs = self.get("tiering.cutoffs", None)
            if not cutoffs or len(cutoffs) != 2:
                raise ConfigError(
                    "tiering.strategy=fixed 时必须显式提供两个切点 tiering.cutoffs: [低中边界, 中高边界]。"
                    "本框架不提供默认阈值，因为风险分级的切点应由业务定义、历史命中率或"
                    "成本函数决定，而不是由程序猜"
                )
            lo, hi = float(cutoffs[0]), float(cutoffs[1])
            if not (0 < lo < hi < 1):
                raise ConfigError(f"tiering.cutoffs 必须满足 0 < 低 < 高 < 1，当前 {cutoffs}")
        if t_strategy == "quantile":
            qs = self.get("tiering.quantiles", None)
            if not qs or len(qs) != 2:
                raise ConfigError("tiering.strategy=quantile 时需要两个分位点 tiering.quantiles")
            q1, q2 = float(qs[0]), float(qs[1])
            if not (0 < q1 < q2 < 1):
                raise ConfigError(f"tiering.quantiles 必须满足 0 < q1 < q2 < 1，当前 {qs}")
        labels = self.get("tiering.labels", [])
        if len(labels) != 3:
            raise ConfigError("tiering.labels 必须是三个档位的名称")
        if self.get("tiering.fit_on", "val") not in {"val", "train"}:
            raise ConfigError("tiering.fit_on 只能是 val 或 train")

        # ---- vulnerability overlay
        if self.get("vulnerability_overlay.enabled", False):
            matrix = self.get("vulnerability_overlay.matrix", None)
            if not isinstance(matrix, dict):
                raise ConfigError("vulnerability_overlay.matrix 必须是映射")
            for risk in labels:
                if risk not in matrix:
                    raise ConfigError(f"vulnerability_overlay.matrix 缺少风险档 {risk!r}")
                if set(matrix[risk]) != {"low", "medium", "high"}:
                    raise ConfigError(
                        f"vulnerability_overlay.matrix[{risk!r}] 需覆盖 low/medium/high 三个脆弱档"
                    )

        # ---- leakage
        if not isinstance(self.get("leakage.forbidden_patterns", []), list):
            raise ConfigError("leakage.forbidden_patterns 必须是列表")

        # ---- gis
        allowed_exports = {"csv", "geojson", "gpkg"}
        exports = set(self.get("gis.exports", []))
        if not exports <= allowed_exports:
            raise ConfigError(f"gis.exports 只支持 {sorted(allowed_exports)}，当前 {sorted(exports)}")

        self.warnings = w

    # ------------------------------------------------------------------ 落盘
    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
        return p
