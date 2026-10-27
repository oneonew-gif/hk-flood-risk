# 01 · 技术架构

> 本文件描述当前仓库实际代码结构（2026-10-27 核查），与 `README.md` 目录说明一致。

---

## 1. 顶层结构

```
hk-flood-risk/
├─ configs/                  # 配置：base.yaml 是所有默认值的唯一来源
│  ├─ base.yaml              #   任务/数据/特征/切分/训练/评价/校准/分级/泄漏/GIS/输出
│  ├─ synthetic_dynamic.yaml #   合成数据（动态模式）冒烟配置
│  └─ synthetic_static.yaml  #   合成数据（静态模式）冒烟配置
├─ src/floodrisk/            # 建模框架（Python 包，pip install -e . 可编辑安装）
├─ scripts/                  # 实验与数据处理脚本（每个正式实验一个独立脚本）
├─ tests/                    # pytest，167 项（2026-10-27 collect 实测）
├─ data/                     # 数据版本目录（见 02_DATA_INVENTORY.md）
├─ outputs/                  # 实验产物（每次运行独立目录，不覆盖）
├─ reports/                  # 研究与决策报告（29 份 md）
├─ hko_api/                  # 产品轨道：HKO 实时接入 + Dashboard + 18 区播报
├─ HANDOFF/                  # 本交接包（2026-10-27 新增）
├─ README.md                 # 建模框架使用说明
├─ PROJECT_PRINCIPLES.md     # 双轨开发长期规范 v1.0
├─ ITERATIONS.md             # 迭代记录 #0–#24
├─ CHANGELOG.md              # 数据版本/研究口径变更日志
├─ model_demo.html           # 早期单文件交互演示（独立产物，245 KB）
├─ pyproject.toml / requirements.txt / uv.lock
└─ .gitignore
```

## 2. 建模框架 `src/floodrisk/`（17 个模块）

| 模块 | 职责 | 状态 |
|---|---|---|
| `config.py` | 配置加载/校验/派生量/指纹（`extends` 继承） | ✅ |
| `schema.py` | 字段契约（规范名 ↔ 别名，含预报类字段） | ✅ |
| `io.py` | 读入 + 列映射 + 校验报告；**不填缺失值、不改越界值**；重复主键中止 | ✅ |
| `synthetic.py` | 合成数据生成（仅自测；产出 `*.provenance.json`，SYNTHETIC 不可引用） | ✅ |
| `splits.py` | temporal / spatial_block / spacetime / random 四种切分 + 标签窗口 purge + 同源 CV；无坐标时空间切分**报错不退化** | ✅ |
| `leakage.py` | 三道泄漏闸门（特征黑名单 / 按时间轴的 lag·rolling 校验 / 切分不相交）+ 单变量 AUC 扫描（**仅限训练段**，`scan_idx` 必填） | ✅ |
| `features.py` | 按时间轴 lag/rolling、预报窗口聚合、静态直通；rolling 族运行时复核 | ✅ |
| `train.py` | XGBoost 训练、早停、不平衡处理、特征重要性 | ✅ |
| `tuning.py` | 随机搜索（折与主切分同源，含 purge）；**默认关闭，需批准** | ✅ |
| `evaluate.py` | 7 指标 + 混淆矩阵 + 阈值扫描 + ROC/PR 曲线 | ✅ |
| `calibrate.py` | none / platt / isotonic + 可靠性曲线 + ECE | ✅ |
| `pipeline.py` | 端到端串联 + manifest + provenance 闸门 + 标签口径闸门（`label_horizon_verified`） | ✅ |
| `realdata.py` | **A 交付真实数据适配层**：`load_delivery`（两遍读防爆内存）、`build_targets`（主目标不位移）、`assign_splits`、`FEATURE_GROUPS`（59 特征）与 `EXCLUDED`（57 列排除清单） | ✅ 正式使用 |
| `models.py` | 算法无关的 `Bundle` joblib 契约；`predict_proba` 强制 float64 | ✅ 正式使用 |
| `predict.py` | 批 3：从训练输出目录恢复模型/校准器/配置，对新数据预测 | ✅ |
| `gis_export.py` | 批 4：预测 ↔ grid_id 关联导出 GeoJSON（纯 dict/json，不依赖 geopandas） | ✅ |
| `tiering.py` / `vulnerability.py` / `cli.py` | README 中规划的批 3/4 模块 | ⬜ 未建（正式实验走 `scripts/` 路线，见下） |

> 说明：README 目录树是批 1–2 时期的规划；正式真实数据实验并未走 `pipeline.py`，而是走 `scripts/run_*.py` + `realdata.py`/`models.py` 路线。两条路线共用 `splits/leakage/features` 的防泄漏思想，但代码上是独立实现。

## 3. 实验脚本 `scripts/`（16 个）

| 脚本 | 用途 | 产物目录 |
|---|---|---|
| `b_side_repair.py` | R2 修复：HKO 官方 rstorm.dat 重建标签（feasibility/repair/validate 三子命令，seed=42） | `data/delivery_repaired_b1/`、`outputs/repair_b1/` |
| `run_small_experiment.py` / `run_small_experiment_v10_1.py` / `run_small_experiment_repaired_b1.py` | ≤30 万样本轻量实验（v10.0 / v10.1 ERA5 消融 / 修复版） | `outputs/small_experiment_*/` |
| `run_formal_training_repaired_b1.py` | **正式全量训练**（--preflight 六道中止闸门 + 5 模型 + 阈值扫描 + SHAP + GIS） | `outputs/formal_training_repaired_b1_20261026-020814/` |
| `run_supplement_tide_ablation_b1.py` | 去潮位消融 + 一致性 + 动态能力 + 阈值敏感性 + 雨潮复合 | `outputs/supplement_tide_ablation_b1_20261026-024612/` |
| `run_final_verification_spatial_b1.py` | 最终只读核验（聚合 AUC / 雨潮组成 / 空间排名 / GIS 口径） | `outputs/final_verification_spatial_b1_20261026-030632/` |
| `run_paper_package_b1.py` | 论文图表包整理（只读，数值直取冻结 metrics.json） | `outputs/paper_package_b1_20261026-122643/` |
| `chapter4_descriptive_stats.py` / `chapter4_extract_existing_data.py` | 第四章描述性统计与四表数据提取 | `outputs/chapter4_statistics/`、`outputs/chapter4_tables/` |
| `check_delivery_v10_1.py` / `final_pretraining_audit.py` | 交付核查与训练前审计 | `outputs/v10_1_check/`、`outputs/final_audit_20261025/` |
| `build_stpu_base.py` / `audit_t3_quality_20261027.py` | T3 STPU 底表构建与质量审计 | `outputs/t3_stpu_base_20261027/`、`outputs/t3_quality_audit_20261027/` |
| `profile_cv_folds_20261027.py` | 空间 CV 折有效性核验与设计（**未执行 CV 训练**） | `outputs/spatial_cv_design_20261027/` |
| `predict_from_bundle.py` | 独立预测 CLI（bundle + 输入表 → 出分） | — |

## 4. 产品轨道 `hko_api/`

```
hko_api/
├─ hko_fetch.py        # HKO 五源轮询抓取（纯标准库，无密钥）：warnings / rhrread /
│                      #   hourlyRainfall / hhot / nowcast；raw 归档 + 双时间戳 + STALE 降级
├─ prepare_scores.py   # 只读冻结测试期预测 → 5,517 格评分（grid_scores.json/js）
├─ dashboard.html      # 单文件 Dashboard（Leaflet 1.9.4 vendored 离线可用）
├─ vendor/             # leaflet.css / leaflet.js（本地化，无 CDN 依赖）
├─ district/           # T6：18 区气象播报页
│  ├─ index.html / bulletin.js（纯函数规则模板）/ tests/bulletin.test.mjs（node，8 组）
│  ├─ README.md        # 运行/口径/限制/授权声明
│  └─ reports/         # 字段映射、验收记录、GitHub 发布检查清单
├─ reports/            # mvp_readout.md、hko_api_inventory_and_feature_mapping.md
└─ data/
   ├─ raw/             # 原始响应归档（280 MB，.gitignore 排除）
   └─ latest/          # 当前快照 snapshot.json/js + grid_scores.json/js（1.7 MB）
```

## 5. 数据流（研究轨道主线）

```
A 侧交付（data/delivery_v10*, 只读归档）
   │  警告源错误（红色火灾混入）→ D1 审计 → R2 修复
   ▼
scripts/b_side_repair.py ──► data/delivery_repaired_b1/  【冻结·正式训练版本】
   │  （HKO rstorm.dat 官方警告重建 10 个标签列；59 特征与其余列逐行未动）
   ▼
scripts/run_formal_training_repaired_b1.py
   │  train 2018–2021 / val 2022 / test 2023–2026.09（A.split 原划分沿用）
   ▼
outputs/formal_training_repaired_b1_20261026-020814/  【冻结】
   │  5 bundles + metrics.json + predictions_val/test.parquet
   │  + shap_importance.csv + gis/grid_risk_500m.csv + 6 图
   ▼
scripts/run_supplement_tide_ablation_b1.py ──► outputs/supplement_...（no_tide_xgb）
scripts/run_final_verification_spatial_b1.py ──► outputs/final_verification_...（只读核验）
scripts/run_paper_package_b1.py ──► outputs/paper_package_b1_20261026-122643/  【冻结·论文图表包】
   ▼
outputs/paper_tables_fill_20261026/（四表填充 + docx 审核副本）
   ▼
~/Desktop/论文四表填充_20261026.zip（桌面交付副本，见 05 §外部产物）
```

## 6. 数据流（产品轨道）

```
HKO Open Data（data.weather.gov.hk，无需密钥）
   │  hko_fetch.py 轮询（warnings 60s / rhrread 300s / hourlyRainfall 300s / hhot 3600s / nowcast 720s）
   ▼
hko_api/data/raw/{source}/{YYYYMMDD}/  （原始归档 + .meta.json + fetch_log.jsonl）
   ▼
hko_api/data/latest/snapshot.json|js   （归一化快照：obs_time / fetch_time / stale / last_error）

outputs/formal_training_.../predictions_test.parquet（冻结，只读）
   │  prepare_scores.py
   ▼
hko_api/data/latest/grid_scores.json|js（5,517 格测试期/全期双口径评分）

snapshot + grid_scores ──► dashboard.html（三类图层：观测 / 官方预报 / 历史模型评分）
snapshot ──► district/index.html（18 区过去 1h 雨量播报；与风险图层严格分离）
```

## 7. 关键架构决策（详见 05）

- **时间锚**：行时间戳 = 槽开始（预测发起时刻），主目标 `y_future_2h` 不额外位移；位移版仅敏感性分析
- **特征可用性**：59 特征全部为 t 时刻可得或静态不变量；时间窗口/滞后/滚动一律以**步数**为单位
- **防泄漏**：泄漏闸门 + 切分 purge + 单变量 AUC 仅扫训练段；正式训练前 preflight 六道中止闸门
- **无调参**：正式 5 模型均默认超参（`configs/base.yaml` 的 train.params），tuning.enabled=false
- **阈值**：每模型在 val 上 0.01–0.99 扫描 F1 最大点选定；test 不参与选阈；旧 0.30 阈值已弃用
- **校准**：正式训练 `calibration.method: none`（概率尺度失真已披露，输出按「评分」而非「概率」表述）
