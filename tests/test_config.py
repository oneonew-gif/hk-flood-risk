"""配置层测试：默认值来源、派生量、以及"绝不兜底"的三级阈值。"""

from __future__ import annotations

import pytest

from floodrisk.config import Config, ConfigError


def test_base_config_loads(project_root):
    cfg = Config.load(project_root / "configs" / "base.yaml")
    assert cfg.mode == "dynamic"
    assert cfg.is_dynamic is True
    assert cfg.anchor == "issue_time"
    assert cfg.horizon_steps == 2          # 2h / 1h，只作用于标签窗口
    assert cfg.min_feature_lag == 1        # conservative 策略（A 确认约定前的保守值）
    assert cfg.effective_split_strategy() == "temporal"
    assert cfg.get("tiering.quantiles") == [0.70, 0.90]
    assert cfg.get("features.temporal.lags") == [1, 2, 3, 6, 12]


def test_anchor_enum_is_enforced(project_root):
    with pytest.raises(ConfigError, match="issue_time / window_end"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"task": {"anchor": "sometime"}},
        )


def test_convention_requires_anchor_for_interval(project_root):
    """interval 聚合必须声明时间戳锚在时段哪一头，否则任何 lag 都谈不上安全。"""
    raw = Config.load(project_root / "configs" / "base.yaml").to_dict()
    raw["data"]["time_conventions"] = {"rainfall": {"aggregate": "interval"}}  # 故意漏 anchor
    with pytest.raises(ConfigError, match="anchor"):
        Config.from_dict(raw)


def test_convention_aggregate_enum(project_root):
    raw = Config.load(project_root / "configs" / "base.yaml").to_dict()
    raw["data"]["time_conventions"] = {"rainfall": {"aggregate": "vibes"}}
    with pytest.raises(ConfigError, match="interval / instantaneous / static"):
        Config.from_dict(raw)


def test_min_lag_conservative_rejects_lag_zero(project_root):
    """用户决定：A 书面确认时间戳约定之前，观测类特征一律 lag >= 1。"""
    with pytest.raises(ConfigError, match="最小合法滞后"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"features": {"temporal": {"lags": [0, 1]}}},
        )


def test_min_lag_derived_allows_lag_zero_when_safe(project_root):
    """end 锚定 + 0 延迟 → lag 0 合法（整点值 = 过去时段累计，发起时刻已观测完）。"""
    cfg = Config.load(
        project_root / "configs" / "base.yaml",
        overrides={
            "features": {"temporal": {"min_lag_policy": "derived", "lags": [0, 1, 2]}}
        },
    )
    assert cfg.min_feature_lag == 0


def test_start_anchored_convention_pushes_min_lag(project_root):
    """时间戳=时段起点 → 值在 +1 步才观测完 → derived 最小滞后应为 1。"""
    cfg = Config.load(
        project_root / "configs" / "base.yaml",
        overrides={
            "data": {"time_conventions": {"rainfall": {"aggregate": "interval", "anchor": "start"}}},
            "features": {"temporal": {"min_lag_policy": "derived"}},
        },
    )
    assert cfg.min_feature_lag == 1


def test_window_end_anchor_requires_horizon_lag(project_root):
    """复盘口径：特征必须整段早于标签窗口打开 → lag >= horizon_steps。"""
    cfg = Config.load(
        project_root / "configs" / "base.yaml",
        overrides={
            "task": {"anchor": "window_end"},
            "features": {"temporal": {"lags": [3, 6]}},
        },
    )
    assert cfg.min_feature_lag == 2
    with pytest.raises(ConfigError, match="最小合法滞后"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={
                "task": {"anchor": "window_end"},
                "features": {"temporal": {"lags": [1, 2]}},
            },
        )


def test_forecast_window_stats_validated(project_root):
    with pytest.raises(ConfigError, match="window_stats"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"features": {"forecast": {"enabled": True, "variables": ["tide_ast"], "window_stats": ["median"]}}},
        )


def test_forecast_requires_variables_when_enabled(project_root):
    with pytest.raises(ConfigError, match="variables"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"features": {"forecast": {"enabled": True, "variables": []}}},
        )


def test_extends_inherits_and_overrides(project_root):
    cfg = Config.load(project_root / "configs" / "synthetic_static.yaml")
    # 继承自 base.yaml
    assert cfg.get("train.params.max_depth") == 4
    # 被覆盖
    assert cfg.mode == "static"
    assert cfg.effective_split_strategy() == "spatial_block"
    assert cfg.get("features.temporal.lags") == []


def test_static_mode_rejects_temporal_lags(project_root):
    with pytest.raises(ConfigError, match="static"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"task": {"mode": "static"}},
        )


def test_static_mode_rejects_window_end_anchor(project_root):
    with pytest.raises(ConfigError, match="anchor"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={
                "task": {"mode": "static", "anchor": "window_end"},
                "features": {"temporal": {"lags": []}},
            },
        )


def test_fixed_tiering_without_cutoffs_fails(project_root):
    """关键设计：不提供默认阈值，缺 cutoffs 必须中止。"""
    with pytest.raises(ConfigError, match="cutoffs"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"tiering": {"strategy": "fixed", "cutoffs": None}},
        )


def test_fixed_tiering_rejects_non_monotonic_cutoffs(project_root):
    with pytest.raises(ConfigError):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"tiering": {"strategy": "fixed", "cutoffs": [0.7, 0.3]}},
        )


def test_fixed_tiering_accepts_explicit_cutoffs(project_root):
    cfg = Config.load(
        project_root / "configs" / "base.yaml",
        overrides={"tiering": {"strategy": "fixed", "cutoffs": [0.22, 0.61]}},
    )
    assert cfg.get("tiering.cutoffs") == [0.22, 0.61]


def test_quantile_tiering_requires_two_points(project_root):
    with pytest.raises(ConfigError, match="quantiles"):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"tiering": {"quantiles": [0.5]}},
        )


def test_invalid_split_ratio_rejected(project_root):
    with pytest.raises(ConfigError):
        Config.load(
            project_root / "configs" / "base.yaml",
            overrides={"split": {"test_size": 0.9, "val_size": 0.9}},
        )


def test_random_split_warns_on_real_data(project_root):
    cfg = Config.load(
        project_root / "configs" / "base.yaml",
        overrides={"split": {"strategy": "random"}},
    )
    assert any("random" in w for w in cfg.warnings)


def test_fingerprint_is_stable_and_sensitive(project_root):
    a = Config.load(project_root / "configs" / "base.yaml")
    b = Config.load(project_root / "configs" / "base.yaml")
    c = Config.load(
        project_root / "configs" / "base.yaml", overrides={"train": {"params": {"max_depth": 5}}}
    )
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_missing_key_raises(project_root):
    cfg = Config.load(project_root / "configs" / "base.yaml")
    with pytest.raises(ConfigError, match="缺少配置项"):
        cfg.get("train.nonexistent_key")
    assert cfg.get("train.nonexistent_key", "fallback") == "fallback"
