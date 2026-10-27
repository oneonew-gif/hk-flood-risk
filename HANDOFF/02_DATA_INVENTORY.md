# 02 · 数据资产清单

> 核查日期：2026-10-27。哈希仅对冻结数据集/关键交付文件计算（`shasum -a 256` 只读实测）；其余记录路径、大小、用途。
> **授权总述**：A 侧交付数据与衍生数据集均为比赛/论文内部使用，**无公开再分发授权**；HKO 实时数据受天文台开放数据条款约束。任何对外再分发都需另行核实许可。

---

## 1. 冻结正式数据（禁止覆盖 / 删除 / 移动）

### 1.1 `data/delivery_repaired_b1/`（166 MB）— 正式训练唯一版本

| 文件 | 大小 | SHA256 | 说明 |
|---|---|---|---|
| `flood_risk_dataset_2h_500m.parquet` | ~166 MB | `1749cc008dc5754184b69f927ee3a4086765ee3dffe4d431be8a617a265cb4e8` ✅ 实测一致 | **2,905,896 行 × 109 列**；5,517 格 × 2h 槽；2018-01-01 ~ 2026-09-30 |
| `sample_frame.parquet` | — | `423d1e853fe2f4d0f247df7aa744c0244bc1add48e80db4941ed792f1f47d79b` ✅ | 抽样框架（35 格/日） |
| `manifest_repair_b1.json` | — | `49dcfdd8f2eb29bd16cfd98c54ad251bbc8d30e4d2e65b995023a175d0f85576` ✅ | 修复全记录：来源/哈希/种子(seed=42)/参数/差异统计 |

- **主键**：`(grid_id, date, slot)` 重复 0；`timestamp = date + 2h·slot`
- **划分**（`split` 列，沿用 A 侧 + B 侧二次切分）：train 2018–2021 = 1,199,472 行 / val 2022 = 209,148 / test 2023-01~2026-09 = 1,496,340；grid-day 跨集合重叠 0；另有 golden_eval 936 行（两个文档化黄金事件，仅敏感性参考）
- **标签**：`flooded_2h`（主目标，不位移）；弱监督 = `is_A` 空间规则 × 官方红/黑警告窗口；正例 196,693（6.77%）；56 个官方事件日（红 42 + 黑 14）；**320 个旧假事件日已改判非事件日并受限重抽**（A 类占比偏差 55.5%，已披露）
- **特征**：109 列中 59 个模型特征（`realdata.py` 注册表）+ 57 列 EXCLUDED（标识/标签/规则内部列）
- **cv_fold**：{-1, 0..17}；-1 = 41,964 行（44 个无区网格 41,028 + golden 936）

### 1.2 字段分组（109 列概览）

| 组 | 列数 | 举例 |
|---|---|---|
| 标识/标签/划分 | ~20 | grid_id, date, slot, flooded*, label_*, split, cv_fold, is_A* |
| 时间 | 5 | month, hour_of_day, hour_sin, hour_cos, warn_level |
| 潮位 | 6 | tide_station, tide_pred_max/mean_m, surge_max_m, tide_n_hours, dist_to_tide_m |
| ERA5 降雨 | 9 | era5_rain_slot_2h/prev3h/…/prev48h, max1h_prev6h/24h, api72 |
| POWER 降雨 | 9 | power_rain_* 同上结构 |
| 地形/坐标 | ~18 | i, j, cx_2326, cy_2326, lon, lat, elevation*, slope_deg, tpi_500, hand/twi… |
| 土地覆盖/利用 | ~16 | esa_lc_*（10 类）, impervious_frac, lu_* |
| 人口/社会 | 6 | pop_worldpop, pop_density_gis, elderly_frac_gis, median_income_gis, pop_density_dcd… |
| 排水/先验 | 5 | dist_to_pump, dist_to_outfall, dist_to_blackspot_m, dist_to_lowlying_m, flood_prone_flag, coastal_lowlying_flag |
| 1km 水文派生 | 8 | hydro1km_* |

## 2. A 侧历史交付（只读归档，不用于正式训练，不得删除）

| 目录 | 大小 | 说明 |
|---|---|---|
| `data/delivery_v10_20261025/` | 401 MB | **A 侧最终交付**。主表 385 MB（11,841,576 行 × 109 列，SHA256 `9f580846…dea7822`，与 v10.1 逐字节一致）；含 `diagnostic_full_days.parquet`（24 MB，诊断集，隔离只读）、`grid_500m_static.csv`（5,517×66）、`grid_500m_to_stpu.csv`、`grid_500m_stpu_overlap.csv`、`golden_grid_map_500m.csv`（87 行）、6 个 QC JSON、README 等 4 份文档。**注意：此版标签含 320 个假事件日（火灾警告混入），仅作历史留档** |
| `data/delivery_v10_1/` | 378 MB | v10.1（ERA5 补齐版），结构与上同 |
| `data/delivery_v10/` | 259 MB | v10.0 首版 |

## 3. STPU 映射与人口属性（在 delivery_v10_1 与 v10_20261025 内）

| 文件 | 规模 | 状态 |
|---|---|---|
| `grid_500m_to_stpu.csv` | 5,517 → 207 STPU（pip 4,334 + area_max 1,183），448 KB | 待验证（QC 通过但未入冻结训练集；T3 已核） |
| `grid_500m_stpu_overlap.csv` | 7,703 行 × 211 STPU，151 KB，**4 位小数面积权重** | 待验证；聚合一律用此长表（3 位主映射表最大偏差 0.0316） |
| 207 vs 211 差异 | 4 个 STPU（122/214/228/231）仅各有 4 个重叠网格，从未成为主映射 | 已定位，两套口径不矛盾 |

## 4. 合成数据 `data/synthetic/`（1.8 MB）

- `panel_synthetic.csv` / `static_synthetic.csv` / `grids_synthetic.geojson` + 两个 `*.provenance.json`
- **SYNTHETIC，坐标在几内亚湾，与香港无关；任何指标/图件禁止引用**，仅证明程序能跑通

## 5. 实时数据 `hko_api/data/`（与论文历史数据完全隔离）

| 子目录 | 大小 | 说明 |
|---|---|---|
| `raw/` | 280 MB（~2,800+ 文件，nowcast 为主） | 原始响应按 `{source}/{YYYYMMDD}/` 归档 + `.meta.json`（URL/HTTP 状态/双时间戳）+ `fetch_log.jsonl`；**.gitignore 已排除** |
| `latest/` | 1.7 MB | `snapshot.json|js`（5 源归一化快照）+ `grid_scores.json|js`（5,517 格历史评分，源自冻结预测） |

- 5 源：warnings / rhrread / hourlyRainfall / hhot / nowcast；每源 obs_time（官方产品时刻）与 fetch_time（管道获取时刻）分离；失败保留上次成功快照 + `stale:true`
- **约束**：实时数据不得写入任何 `data/delivery_*` 或 `outputs/` 冻结目录；不得作为冻结模型输入对外展示「实时水浸概率」

## 6. 关键实验产物数据（outputs/ 内，冻结）

| 路径 | 大小 | 说明 |
|---|---|---|
| `outputs/formal_training_repaired_b1_20261026-020814/` | 51 MB | 正式训练全套（bundles × 5、predictions_test.parquet 42 MB / 1,496,340 行、predictions_val.parquet 6.8 MB / 209,148 行、metrics.json、shap_importance.csv、gis/grid_risk_500m.csv、阈值扫描 × 5、图 × 6、results_report.md） |
| `outputs/supplement_tide_ablation_b1_20261026-024612/` | 11 MB | no_tide_xgb bundle、predictions_test_no_tide.parquet、雨潮复合/沿海分组/阈值敏感性表 |
| `outputs/paper_package_b1_20261026-122643/` | 3.4 MB | 论文图表包 18 文件 + `MANIFEST.sha256`（逐文件哈希，交接时已核验存在） |
| `outputs/paper_package_b1_20261026-122643 2/` + `.zip` | 3.4 MB / 2.7 MB | **冗余副本，唯一权威副本待确认（checklist R7）** |
| `outputs/zis5HLyp` | 2.8 MB | 无名 zip（2026-10-26 12:50，疑似分享临时包）；**用途待验证** |

## 7. 仓库外协作材料

正式数据、模型 bundle、论文图表包和四表交付材料不纳入 Git。项目负责人应通过团队共享存储向获授权的协作者提供，并同时提供版本号、SHA256 与只读/可修改说明；仓库文档不记录任何个人电脑上的绝对路径。

## 8. 缺失/未交付（不得伪造）

- 500 m 网格真实边界几何（全库仅合成 geojson；地图按格中心点展示）
- STPU 官方边界几何；18 区边界几何
- `census2021_stpu_units.csv`（2021 普查 STPU 原始底表，引用文件名不存在）
- 渠务署（DSD）黑点官方名单；社区（区议会选区）边界映射
- SVI 计算方法说明（A 方未提供）
- 2021-12-30/31 两天警告根因（待 A 方 `parsed_warnings.json`）
