"""字段契约（schema）。

为什么需要这一层
----------------
A 同学交付的数据列名未知，可能叫 ``RF_1h``、也可能是 ``rain_mm``。若把列名写死
在代码里，换数据就要改代码。这里定义"规范名 → 别名集合"，再由配置里的
``data.columns`` 做显式覆盖，形成两级缓冲。

硬约束只有两个：``grid_id`` 与 ``flood_label``。其余字段全部可选 —— 缺了不报错，
只记录到 validation_report，让你能先跑最小可行版本，再逐步加特征。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # key | time | feature | target | vulnerability | group
    aliases: tuple[str, ...] = ()
    unit: str | None = None
    valid_range: tuple[float, float] | None = None
    note: str = ""

    @property
    def is_feature(self) -> bool:
        return self.kind == "feature"


CANONICAL: dict[str, FieldSpec] = {
    # ---------------- 主键 ----------------
    "grid_id": FieldSpec(
        name="grid_id",
        kind="key",
        aliases=("grid", "gridid", "cell_id", "cellid", "gid", "网格编号"),
        note="社区网格唯一标识，GIS 关联的锚点，不可缺",
    ),
    "timestamp": FieldSpec(
        name="timestamp",
        kind="time",
        aliases=("time", "datetime", "date_time", "ts", "obs_time", "时间"),
        note="dynamic 模式必需；static 模式忽略",
    ),
    # ---------------- 物理致灾因子 ----------------
    "rainfall": FieldSpec(
        name="rainfall",
        kind="feature",
        aliases=("rain", "rain_mm", "rainfall_mm", "rf", "rf_1h", "precip", "precipitation"),
        unit="mm",
        valid_range=(0.0, 800.0),
        note="时段累计雨量",
    ),
    "tide": FieldSpec(
        name="tide",
        kind="feature",
        aliases=("tide_m", "tide_level", "tide_mcd", "sea_level", "water_level"),
        unit="mCD",
        valid_range=(-1.5, 6.0),
        note="实测潮位（含风暴增水），观测类特征",
    ),
    "tide_ast": FieldSpec(
        name="tide_ast",
        kind="forecast",
        aliases=("astro_tide", "tide_astronomical", "astronomical_tide", "tide_pred"),
        unit="mCD",
        valid_range=(-1.5, 6.0),
        note=(
            "天文潮（确定性预报量）：任意未来时刻的值在发起时刻即可精确计算，"
            "属于合法的'未来特征'，是潮汐顶托叙事的关键变量。"
            "不直接入模，经 features.forecast 对 [t, t+horizon) 窗口聚合后使用"
        ),
    ),
    "wind_speed": FieldSpec(
        name="wind_speed",
        kind="feature",
        aliases=("wind", "wind_ms", "ws", "windspeed", "wind_speed_ms"),
        unit="m/s",
        valid_range=(0.0, 90.0),
        note="风速，风暴潮与海浪的驱动量",
    ),
    "elevation": FieldSpec(
        name="elevation",
        kind="feature",
        aliases=("elev", "dem", "elev_m", "height_m", "altitude"),
        unit="m",
        valid_range=(-10.0, 1000.0),
        note="地面高程，静态地形特征",
    ),
    "slope": FieldSpec(
        name="slope",
        kind="feature",
        aliases=("slope_deg", "slope_degree", "gradient"),
        unit="deg",
        valid_range=(0.0, 90.0),
        note="坡度，影响汇流速度",
    ),
    "drainage_capacity": FieldSpec(
        name="drainage_capacity",
        kind="feature",
        aliases=("drain_capacity", "drainage_design_capacity", "drainage_std"),
        unit="mm/h",
        valid_range=(0.0, 500.0),
        note=(
            "排水设计能力（静态规划参数）。注意：只有'设计标准'可用，"
            "'事后清淤/检查记录'属于标签期信息，命中黑名单会被拦截"
        ),
    ),
    # ---------------- 承灾体（默认不进模型） ----------------
    "population": FieldSpec(
        name="population",
        kind="vulnerability",
        aliases=("pop", "spop", "population_count", "residents"),
        unit="count",
        valid_range=(0.0, 500000.0),
        note="网格常住人口。默认只用于叠加分析，不作为水浸物理成因",
    ),
    "vulnerability_score": FieldSpec(
        name="vulnerability_score",
        kind="vulnerability",
        aliases=("vuln", "vuln_score", "vulnerability", "vulnerability_index"),
        unit="0-1",
        valid_range=(0.0, 1.0),
        note="综合脆弱度（人口结构、长者比例、地下设施等）。默认只用于叠加分析",
    ),
    # ---------------- 分组 / 几何辅助 ----------------
    "spatial_block_id": FieldSpec(
        name="spatial_block_id",
        kind="group",
        aliases=("block_id", "district", "district_id", "admin_id", "subzone"),
        note="空间分块标识（行政区或聚类块），空间切分与分组 CV 的依据",
    ),
    "lon": FieldSpec(name="lon", kind="group", aliases=("longitude", "x", "经度")),
    "lat": FieldSpec(name="lat", kind="group", aliases=("latitude", "y", "纬度")),
    "x_km": FieldSpec(name="x_km", kind="group", aliases=("x_coord", "easting")),
    "y_km": FieldSpec(name="y_km", kind="group", aliases=("y_coord", "northing")),
    # ---------------- 目标 ----------------
    "flood_label": FieldSpec(
        name="flood_label",
        kind="target",
        aliases=("label", "flood", "is_flood", "flooded", "target", "y", "水浸标签"),
        note="二分类标签，取值必须为 0/1；口径由使用者定义，本框架不代劳",
    ),
}

REQUIRED_ALWAYS: tuple[str, ...] = ("grid_id", "flood_label")

FEATURE_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in CANONICAL.items() if spec.kind == "feature"
)
FORECAST_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in CANONICAL.items() if spec.kind == "forecast"
)
VULNERABILITY_FIELDS: tuple[str, ...] = tuple(
    name for name, spec in CANONICAL.items() if spec.kind == "vulnerability"
)


def required_fields(mode: str, require_label: bool = True) -> tuple[str, ...]:
    """mode 必需字段。``require_label=False`` 用于预测场景（无标签数据）。"""
    base = ["grid_id"] + (["timestamp"] if mode == "dynamic" else [])
    if require_label:
        base.append("flood_label")
    return tuple(base)


def _norm(name: str) -> str:
    return str(name).strip().lower()


@dataclass
class ResolvedColumns:
    """列名解析结果。"""

    mapping: dict[str, str] = field(default_factory=dict)
    """canonical -> 数据中的实际列名"""

    unresolved: dict[str, str] = field(default_factory=dict)
    """canonical -> 未解析原因"""

    extra: list[str] = field(default_factory=list)
    """数据里存在但未被契约识别、也未在 columns 中声明的列"""

    notes: list[str] = field(default_factory=list)

    def actual(self, canonical: str) -> str | None:
        return self.mapping.get(canonical)

    def has(self, canonical: str) -> bool:
        return canonical in self.mapping


def resolve_columns(
    columns: Iterable[str],
    overrides: dict[str, str] | None = None,
    *,
    mode: str = "dynamic",
) -> ResolvedColumns:
    """把数据列名映射到规范名。

    优先级：显式覆盖 > 别名匹配 > 规范名同名匹配。
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    cols = list(columns)
    norm_to_actual = {_norm(c): c for c in cols}
    used: set[str] = set()
    out = ResolvedColumns()

    for canonical, spec in CANONICAL.items():
        # 1) 显式覆盖
        if canonical in overrides:
            want = overrides[canonical]
            if want in cols:
                out.mapping[canonical] = want
                used.add(want)
                continue
            hit = norm_to_actual.get(_norm(want))
            if hit is not None:
                out.mapping[canonical] = hit
                used.add(hit)
                out.notes.append(f"覆盖列 {want!r} 按大小写不敏感匹配到 {hit!r}")
                continue
            out.unresolved[canonical] = f"配置中指定的列 {want!r} 不在数据里"
            continue

        # 2) 别名 / 规范名同名
        candidates = (canonical, *spec.aliases)
        hit = None
        for cand in candidates:
            actual = norm_to_actual.get(_norm(cand))
            if actual is not None and actual not in used:
                hit = actual
                break
        if hit is not None:
            out.mapping[canonical] = hit
            used.add(hit)
        else:
            reason = "数据中未找到该列（可选字段，不影响运行）"
            if canonical in required_fields(mode):
                reason = f"{mode} 模式的必需字段，缺失将中止"
            out.unresolved[canonical] = reason

    out.extra = [c for c in cols if c not in used]
    return out
