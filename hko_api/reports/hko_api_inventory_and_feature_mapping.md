# HKO Open Data API 接入 — 接口清单与特征映射（审核稿）

**日期**：2026-10-26
**状态**：待审核 —— 批准后才进入代码开发
**本轮范围**：仅 API 可行性验证 + 独立数据接入模块 + HTML 演示 + 可行性报告。
**明确不做**：不修改现有模型、训练数据、标签、阈值、GIS 历史结果；不将实时 HKO 数据输入 XGBoost；不展示任何未经验证的未来 2 小时水浸概率。
**文件落位**：所有新文件放独立目录 `hko_api/`，不触碰 `src/floodrisk/`、`scripts/`、`data/`、`outputs/`、`reports/` 既有内容。

核验方法：官方《HKO Open Data API Documentation》v1.13（2025-09，48 页 PDF）+ **2026-10-26 12:00–12:35 HKT 全部关键端点实测**（curl 直连，无 API key、无注册，均返回真实数据）。

---

## 1. 实测核验的端点清单

| # | 数据 | 端点 | 格式 | 更新频率/延迟 | 单位 | 空间覆盖 | 实测 |
|---|---|---|---|---|---|---|---|
| 1 | **天气警告摘要**（暴雨黄/红/黑、WFNTSA 水浸特别提示、雷暴、台风信号、山泥倾泻等） | `https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=warnsum&lang=en` | JSON | 分钟级，事件驱动 | — | 全港 | ✅ 当前无警告返回 `{}`；文档含 WRAINA/WRAINR/WRAINB、issueTime/expireTime/actionCode |
| 2 | **天气警告信息**（#1 的详版） | `...weather.php?dataType=warningInfo&lang=en` | JSON | 同上 | — | 全港 | ✅ `{}` |
| 3 | **各区过去 1 小时最大雨量** | `...weather.php?dataType=rhrread&lang=en` | JSON | ~10 min（实测 updateTime 12:02） | mm | 18 区 + 温度/湿度/紫外线 | ✅ |
| 4 | **自动气象站过去 1 小时雨量** | `...weatherAPI/opendata/hourlyRainfall.php?lang=en` | JSON | 逐时，带 `obsTime`（实测 12:15） | mm | 全港 AWS 站网（RF/N 系列站点） | ✅ |
| 5 | **逐时天文潮高度** | `...opendata/opendata.php?dataType=HHOT&station={ST}&year={Y}&rformat=csv` | CSV | 确定性天文预报，可整年预取 | m | 14 站：QUB(鲗鱼涌)、CLK、CCH、KLW、TBT、WAG、TPK 等 | ✅ QUB 2026 全年 |
| 6 | 高低潮时刻与高度 | `...opendata.php?dataType=HLT...` | CSV/JSON | 同上 | m | 14 站 | 文档 |
| 7 | **网格降雨临近预报（nowcast）** | `https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv` | CSV | 每 12 min；未来 2 h、半小时步长 | mm/30min | 121×121 网格（实测 lat 21.328–23.487、lon 112.956–115.291，步长 ~0.018°×0.02°≈2 km）；香港 bbox 内约 900 格 × 4 步 = 3,600 行/次 | ✅ 58,564 行 |
| 8 | 本地天气预报 / 9 天预报 / 特别天气提示 | `weather.php?dataType=flw / fnd / swt` | JSON | 小时级/日级 | — | 全港 | 文档 |
| 9 | 闪电计数 | `opendata.php?dataType=LHL...` | CSV/JSON | 逐时 | 次 | 全港分区 | 文档 |
| 10 | 日总雨量/日均风速（历史口径） | `...weatherAPI/cis/csvfile/{ST}/{Y}/daily_{ST}_RF_{Y}.csv` | CSV | 日更 | mm / m/s | 21 站 | ✅（训练历史已用） |

官方文档 PDF：`https://data.weather.gov.hk/weatherAPI/doc/HKO_Open_Data_API_Documentation.pdf`

### 实测发现的关键口径问题（与 A 方清单的差异）

1. **HHOT 是天文潮（调和预报），不是实测潮位**。A 方《实时数据源与API适配清单》§1 称 HHOT 为"实测/预报"，实测确认 HHOT 仅有天文潮预报列。实测潮位（surge 所需）不在官方 Open Data API 中——但 `surge_max_m` 本就不在 59 特征内，**不影响本轮**。
2. **HHOT 列 01–24 为小时末标注**（列 01 = 当日 01:00 潮位），槽窗口对齐时须减 1 小时（A 方 §2.1 已警告，实测确认）。
3. **nowcast 网格分辨率约 2 km**（0.018°×0.02°），A 方文档写"约 1 km"偏细；且网格域远大于香港（含珠江西侧），必须按 bbox 裁剪。官方标注数据为 provisional（临时数据）。
4. **nowcast 官方不提供历史归档**，只能从接入日起滚动积累——这是未来动态评分验证的最大硬缺口。
5. **hourlyRainfall.php 只返回最新 1 小时快照**，无历史序列接口。prev24h/prev48h/api72 等特征必须由本模块**逐时轮询入库、自行积累**（见 §4 冷启动）。
6. 文档 v1.13 中 HHOT 的 `year` 参数接受值仍写 "2022–2024"，实测 2026 可用（文档滞后）。
7. 调用无需鉴权；使用条款要求合理用量，适配器将内置限流与缓存。

---

## 2. dual_xgb 59 特征 → API 字段映射表

口径约定：预测时刻 = 槽开始 t（主实验"槽开始锚定、不额外位移"），59 特征全部为 t 时刻可得或静态不变量。特征名单与 [realdata.py](../../src/floodrisk/realdata.py) 逐字核对。

### 类别一：可直接获取 / 本地计算（4 个）

| 特征 | 实时来源 | 加工 |
|---|---|---|
| `month` | 本地时钟（HKT） | 无需 API |
| `hour_sin` / `hour_cos` | 本地时钟 | 三角变换 |
| `tide_n_hours` | HHOT | 槽内有效潮位小时数 0–2（结构量，README 口径） |

### 类别二：需要时间聚合或空间插值（18 个）

| 特征 | 历史源（训练） | 实时等价物 | 所需加工 |
|---|---|---|---|
| `tide_pred_max_m`、`tide_pred_mean_m` | 05b 调和分析 | **HHOT 逐时天文潮**（确定性预报，可提前计算） | 仅槽内窗口统计（max/mean），无空间处理；单站缺行用邻近站兜底 |
| `power_rain_prev3h/6h/12h/24h/48h_mm`（5） | NASA POWER 0.5° 逐时 | **hourlyRainfall.php 站点小时雨量（自行积累序列）** | 站点→500 m 插值 + 窗口累计 |
| `power_rain_max1h_prev6h/24h_mm`（2） | 同上 | 同上 | 同上取 max |
| `power_api72_mm`（1） | 同上 | 同上 | 72 h 累计 |
| `era5_rain_prev3h/6h/12h/24h/48h_mm`（5） | ERA5-Land | 同上（与 POWER 共用一路 HKO 观测） | 同上 |
| `era5_rain_max1h_prev6h/24h_mm`（2） | 同上 | 同上 | 同上 |
| `era5_api72_mm`（1） | 同上 | 同上 | 同上 |

> **口径警告（必须写入可行性报告）**：POWER/ERA5 为再分析产品（延迟数天），其原口径**不可实时获取**。类别二的 16 个雨特征实时值来自 HKO 站点观测，与训练分布存在系统性差异（再分析平滑 vs 站点稀疏）——这是部署期已知 distribution shift，须并行记录两路输入做偏差监控。未来若做 2 h 动态评分，槽内未来降雨需 **nowcast**（网格→500 m 面积加权、4 个半小时步长聚合）。

### 类别三：需要其他数据源（本地静态文件，37 个）

直接读交付数据的静态表（固定文件，无需实时 API）：

- `dist_to_tide_m`（静态最近潮位站距离）
- 地形 10：`elevation, elevation_min, elevation_max, elevation_std, slope_deg, elv_rel_mean, tpi_500, roughness_500, land_frac, dem_n`
- 地表覆盖/不透水面 17：`esa_lc_*`（9 类 + pixels/valid_frac）、`impervious_frac`、`lu_*`（4）、`landuse_class`
- 排水设施 2：`dist_to_pump, dist_to_outfall`
- 1 km 水文派生 7：`hydro1km_*`

### 类别四：目前无法实时获取（0 个）

无。但附两条口径级限制：
1. 类别二 16 个雨特征的**原口径**（ERA5/POWER）不可实时，须换 HKO 源（见上警告）；
2. nowcast 无历史归档 →"以预报为输入"的模型表现无法回测，只能从接入日起积累。

**合计 4 + 18 + 37 = 59 ✅**

---

## 3. 拟开发内容（待批准后执行）

```
hko_api/                        # 独立目录，不依赖 src/floodrisk
├── hko_fetch.py                # 数据获取模块（纯标准库 urllib，无新依赖）
│   ├─ 5 个 fetcher：warnings(warnsum+warningInfo) / rhrread / hourlyRainfall / HHOT / nowcast
│   ├─ 轮询节奏：warnings 60s、rhrread 5min、hourlyRainfall 5min、nowcast 12min、HHOT 60min
│   ├─ 每源节流 + 超时 15s + 2 次指数退避重试
│   └─ 异常处理：网络失败/HTTP 非 200/JSON 解析失败/字段缺失 → 记录并保留上一份好快照，不崩溃
├── run_fetch.py                # CLI：--once 单次验证 / --loop 持续积累
├── demo.html                   # 单文件演示页（双击离线可用，快照经 snapshot.js 注入）
├── data/
│   ├── raw/{source}/{YYYYMMDD}/   # 原始响应原样保存 + fetch_log.jsonl（fetch_time/obs_time/HTTP 状态/STALE 标记）
│   └── latest/                    # 归一化快照 snapshot.js + 各源 JSON
└── reports/
    ├── hko_api_inventory_and_feature_mapping.md   # 本审核稿
    └── api_feasibility_report.md                  # 交付 7：可行性报告
```

- **存储原则**：原始 JSON/CSV 原样落盘（provenance 可追溯）；同时记录获取时间（fetch_time）与数据观测时间（obsTime/updateTime/nowcast 更新时刻）。
- **过期判定（STALE 阈值，仅标记不删除）**：warnings 15 min、rhrread 30 min、hourlyRainfall 90 min、nowcast 30 min、HHOT 24 h；demo 页醒目标注每源的更新时间与 STALE 状态。
- **demo.html 内容**：当前生效警告（暴雨黄/红/黑分色、WFNTSA、雷暴、台风信号）、18 区过去 1 小时雨量、AWS 站点雨量 Top 列表、鲗鱼涌（QUB）未来潮位曲线、每源更新时间条。页面固定声明：**"本页仅展示 HKO 官方实况数据，不含任何水浸概率预测或模型输出"**。
- **验证方式**：`--once` 模式实测一轮 5 源抓取；人为断网/改错 URL 演示异常路径；STALE 演示。

---

## 4. 若未来要做真正的 2 小时动态风险评分 —— 缺口清单（写入交付 7）

1. **特征口径差**：训练用 ERA5/POWER 再分析，实时只有 HKO 观测/nowcast → 需从接入日起并行记录两路，量化系统性偏差（论文局限章节的实证支撑）。
2. **冷启动**：prev24h/prev48h/api72 需连续入库 72 h 后才可计算；日雨量 CSV 仅能粗兜底且槽口径不同。
3. **nowcast 无历史归档**：无法回测"以 nowcast 为输入的模型表现"；须积累 3–6 个月"预报 vs 实际"对照样本后才可下动态结论。
4. **空间降尺度未验证**：nowcast ~2 km → 500 m 网格重采样误差、AWS 站点 → 500 m 插值误差（留站交叉验证）均未评估。
5. **槽对齐规则**：nowcast 发布时刻 + 前瞻步长 与建模槽 `[2s, 2s+2)` 的归属规则（12 min 发布延迟导致的边界回退）需定义并记录偏移。
6. **弱监督标签根本约束**：即使输入齐全，模型输出仍是弱监督标签下的探索性风险排序，不是已验证的真实水浸概率——任何动态评分产品的文案必须保留此披露。
7. **校准失效风险**：当前阈值与校准基于训练期特征分布，输入分布漂移后需重新校准，且测试集不得参与。
8. **本轮边界**：上述全部属于未来工作，需另行批准；本轮不调用任何模型 bundle。

---

## 5. 请审核的决策点

1. 端点清单与 59 特征映射分类（§1、§2）是否认可？
2. 模块结构、轮询节奏、STALE 阈值（§3）是否认可？默认只启用 rhrread/warnings/hourlyRainfall/HHOT 四源 + nowcast 采样归档（每 12 min 一份原始 CSV 约 1.4 MB/天 → 建议降频或裁剪 HK bbox 后归档，二选一请指定）。
3. demo 页是否需要中文界面（默认中英双语标注、界面中文）？
