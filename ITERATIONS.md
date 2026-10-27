# 迭代记录

本文件由 AI 在每次完成符合下述定义的修改后**自动追加**，无需人工提醒。

---

## 什么算一次迭代（约定）

**一次迭代 = 一个可命名的变化单元**：能用一句话说清"改了什么、为什么、结果如何"。

判断标准：一个月后回看，如果这条改动会影响你理解"当时模型为什么是这个表现"，它就值得记。

### 记录
- 完成或推进某个明确功能 / 批次（如批 3 的脆弱性叠加）
- 改变模型行为：特征增删、标签口径、阈值策略、切分方式、调参空间
- 改变验证方式：新指标、新评估流程、泄漏闸门调整
- 影响结果正确性的 bug 修复
- 数据管道 / 配置结构变化

### 不记录
- typo、格式、注释、重命名
- 调试过程中的试错、单纯重跑实验
- 不改变行为的纯重构（跨模块的备注一笔即可）

**反向测试**：无法一句话命名的改动——要么太小不值得记，要么太杂应该拆开。

## 记录格式

```
### #编号 · 日期 · 一句话标题
- 意图：为什么改
- 改动：涉及模块/文件
- 验证：pytest 结果 / 关键指标变化
- 遗留：已知问题、下一步（无则省略）
```

---

## 迭代日志

### #0 · 2026-10-22 · 基线快照（此前的修改不追溯）
- 状态：批 1（数据层 + 切分 + 泄漏闸门，按"预测发起时刻"口径重构）+ 批 2（特征 / 训练 / 调参 / 评价 / 校准 / 端到端流水线）已完成；批 3–4 未开始
- 验证：`pytest` 通过；`outputs/batch1_validation_report.json` 已生成；`outputs/` 下有 20261022 两次完整流水线运行
- 说明：本文件于今日建立，作为迭代记录的起点

### #1 · 2026-10-23 · 三处硬性防线修复：provenance 默认反转、val 单类别拦截、rolling 运行时复核
- 意图：修复自审发现的 E1/E2/E3——来源不明的数据此前会被默认标记为可引用的 REAL_DATA（C 类红线）；val 段单一类别不在切分阶段拦截，会在早停/阈值/校准处以难懂形式爆掉；运行时特征复核只抽查 lag 族，预测主力 rolling 族无兜底
- 改动：
  - `pipeline.py` 新增 `_provenance_badge`：仅显式 `REAL` 才 `citable=True` 且 banner 为 `REAL_DATA`；`UNKNOWN` 时 banner 为 `PROVENANCE_UNKNOWN` 且不可引用；复核失败报错文案补上 rolling 项
  - `io.py` 校验报告对 UNKNOWN 来源追加"默认不可引用"及补救方式说明
  - `splits.py` `_assert_usable` 的为空/单一类别检查从 train+test 扩到 train+val+test
  - `features.py` `audit_built_features` 新增 rolling 族复核（调用 `leakage.check_time_rolling`，此前仅单元测试覆盖），audit 结果新增 `rolling_features_checked` / `rolling_failures`，`passed` 纳入 rolling 失败
  - 测试：`test_pipeline.py`（新）覆盖 `_provenance_badge`；`test_schema_io.py` 增加 UNKNOWN 报告断言；`test_splits.py` 增加 val 为空/单类别拦截；`test_features.py` 增加 rolling 复核通过与抓错用例
- 验证：`pytest` 96 项全部通过（此前 91 项）
- 遗留：真实数据首次运行需旁挂 `*.provenance.json` 或配置 `project.provenance_override: REAL`，否则按 UNKNOWN 处理（不可引用）——这是设计代价，已在报告 notes 与 banner 中明示

### #2 · 2026-10-23 · M3 修复：单变量 AUC 泄漏审计限定训练段标签
- 意图：修复自审 M3（P1-03）——`audit_features()` 此前在切分之前对全量数据跑单变量 AUC 扫描，`suspect_features` 经 manifest 回流到人工特征取舍决策（读报告→调 `features.exclude`），构成 val/test 标签的间接泄漏通道（用户授权修复）
- 改动：
  - `leakage.py` `audit_features` 新增必填关键字参数 `scan_idx`，AUC 扫描只用 `dataset.X.iloc[scan_idx]` / `dataset.y[scan_idx]`；报告新增 `auc_scan_scope` 字段明示扫描范围。黑名单检查不使用标签，保持全特征名覆盖
  - `pipeline.py` 泄漏审计从切分前移到切分与特征掩码之后，`scan_idx=tr`（实际参与训练的行）；模块 docstring 顺序同步更新
  - `README.md` 批 1 冒烟示例同步新签名
  - 测试：`test_leakage.py` 新增 `test_train_scoped_auc_scan_is_invariant_to_test_labels`——训练集不变、仅翻转测试集标签时，训练段 AUC 审计结果必须逐特征不变
- 验证：`pytest` 97 项全部通过；敏感性核验：同一份数据翻转测试标签后，全量扫描结果确实变化（泄漏通道非空）、训练段扫描结果不变；端到端 `run_training` 冒烟通过，`auc_scan_scope` 显示"仅传入的 614 行"
- 遗留：审计报告本身仍是"只告警不中止"，suspect_features 的人工判断流程不变；数据质量检查（缺失/越界/重复）在 `io.py` 校验报告中，从不使用标签，不受本次改动影响

### #3 · 2026-10-23 · 批 2.5：预测窗口统一 2h + 标签口径闸门 + 重复值/调参默认收紧 + 非破坏性审计
- 意图：(a) 预测跨度从 3h 改为业务口径 2h，同时用闸门保证"改配置不等于真实标签已是 2h 口径"；(b) 默认不删除真实数据，重复键冲突中止；(c) 默认关闭自动调参；(d) 用独立实现核验 rolling 特征的时间语义
- 改动：
  - `configs/base.yaml`：`task.horizon_hours: 2`；新增 `task.label_horizon_verified: false`；`data.drop_duplicates: false`；`tuning.enabled: false`。**未改** `time_step_hours`、`lags`、`rolling_windows`
  - `configs/synthetic_dynamic.yaml`：`horizon_hours: 2` + `label_horizon_verified: true`（仅声明合成侧自洽）
  - `synthetic.py`：`label_window_steps` 默认 3 → 2
  - `pipeline.py`：新增 `_assert_label_horizon_verified`——非 SYNTHETIC 且未声明口径已验证时**抛错中止**（不是告警后继续）；manifest task 段新增 `horizon_steps` / `horizon_hours` / `label_horizon_verified`；新增 `_window_unit_warnings`（`time_step_hours != 1` 时告警 lags/windows 以步计的实际小时数）
  - `features.py`：报告新增 `step_hours` / `temporal_units: steps`
  - `io.py`：重复值政策重写——完全重复行默认中止（显式开启才删），主键相同但字段冲突一律中止
  - `README.md`：3 处 2h 口径说明，并标注历史指标产生于 3h 窗口、未重跑
  - `.gitignore`：排除 `outputs/`、真实/合成数据、密钥、模型文件、日志、缓存
  - 测试：`test_pipeline.py` 新增真实风格文件名但无来源证明 → UNKNOWN/不可引用；新增口径闸门三例（REAL+未声明 → 中止、声明后放行、SYNTHETIC 不受限）与步数单位告警一例；`test_schema_io.py` 新增完全重复默认中止、主键冲突开去重也中止；同步 6 处测试内 3h 常量到 2h（未放宽断言）
- 验证：`pytest` 139 项全部通过；`outputs/_audit/audit_20261023-005506.json` 五项检查 PASS（AUC 扫描仅训练段 7248/15004 行；rolling 独立实现逐点比对 7638 点 0 不符；打乱行序结果全等；抹掉单点观测后受影响 18 格全为 NaN 且其余 926 格不变；扰动 T 时刻值后 T 及更早特征 0 处变化）
- 遗留：真实数据 2h 标签口径未验证，`label_horizon_verified` 保持 false 会拦截正式训练（需 A 书面确认）；`configs/synthetic_dynamic.yaml` 仍显式 `tuning.enabled: true`（仅冒烟用，非正式调参）；P1-01/P1-07 字段方案待批准；批 3 未启动

### #4 · 2026-10-24 · 批 3 启动：独立预测模块 predict.py
- 意图：训练产物（model.json + config_snapshot.yaml）无法独立复用，需要一个"拿着训练好的模型去预测新数据"的入口，供后续 GIS 导出与外部调用
- 改动：
  - **新建** `src/floodrisk/predict.py`：`predict(run_dir, data_path, cfg=None, output_path=None)` 从训练输出目录恢复模型/校准器/配置，对新数据构造完全相同的特征后输出概率；`PredictionResult` 含 df 与 manifest；`PredictionError` 为独立异常类型
  - **修改** `schema.py`：`required_fields(mode, require_label=True)` 支持无标签预测场景
  - **修改** `io.py`：`load_dataset(..., require_label=False)` 跳过标签校验；`Dataset.y` 无标签时抛 `AttributeError`；`Dataset.meta()` 无标签时 `label_rate=None`；`report["label"]` 无标签时记录说明
  - **修改** `pipeline.py`：`audit_features(ds, cfg, scan_idx=tr)` 从切分前移到切分与特征掩码之后（修复运行时 TypeError）
  - **修改** `__init__.py`：导出 `predict`, `PredictionResult`, `PredictionError`
  - **新建** `tests/test_predict.py`：9 项测试（端到端一致性、无标签预测、缺列报错、校准器回退、provenance 告警、输出列检查、static 模式拦截、run_dir 缺失、model.json 缺失）
  - predict 中 provenance 判定**临时清空**训练配置的 `provenance_override`，使预测数据来源基于输入文件本身而非训练配置
- 验证：`pytest` 148 项全部通过；端到端一致性测试（训练后 predict 同一批数据）概率差异 < 1e-6
- 遗留：`risk_level` 列预留为 None，等 tiering 模块接入；`model_version` 字段为 None，等字段方案批准后填充；static 模式独立预测未实现

### #5 · 2026-10-24 · 批 3 继续：GIS 导出模块 gis_export.py
- 意图：预测概率（表格）需要映射到网格几何（GeoJSON），才能被 GIS 软件或前端地图加载；训练输出未保存 grid_layer 路径，故由调用方显式提供
- 改动：
  - **新建** `src/floodrisk/gis_export.py`：`export_geojson(predictions, grid_layer, output_path, crs, fields, provenance, run_id)` 把预测概率与 grid_layer GeoJSON 通过 grid_id 关联，导出 FeatureCollection；`GISExportError` 为独立异常；不依赖 geopandas，纯 dict/json 生成
  - **修改** `__init__.py`：导出 `export_geojson`, `GISExportError`
  - **新建** `tests/test_gis_export.py`：7 项测试（基础导出、字段保留、缺 grid_id 报错、缺 grid_layer 报错、未知 grid_id 跳过、CSV 输入、timestamp ISO 格式）
- 验证：`pytest` 167 项全部通过
- 遗留：GPKG 导出未实现（需 geopandas）；街道级聚合、交互式地图属 C 类可选；训练流水线未保存 grid_layer 路径，predict → GIS 导出需用户显式提供

### #5 · 2026-10-24 · A 交付 v10 适配层 + 三模型 bundle + 300k 管线实验（目标锚定实测为「槽开始」）
- 意图：交付宽表是 `(grid_id,date,slot)` 三键、混合时间锚，现有 `schema/io/features` 直接套用会缺 timestamp 报错、OOM、并把 ERA5/POWER 两套降雨重复 lag。需要一个独立适配层，并先用 10–30 万行验证整条管线，再决定是否全量训练
- 改动：
  - **新建** `src/floodrisk/realdata.py`：`load_delivery`（arrow 层字典编码 + 两遍读法 filter_keys，避免 5–6 GB 全表入内存）/ `build_targets`（主目标 `y_future_2h` 不位移；对照 `y_future_2h_shift` 严格限同网格+槽位连续，跨网格/缺口置 NaN）/ `assign_splits` / `check_partition`（整数编码 + intersect1d）/ `rule_baseline` / `stratified_sample`（按 年×事件日 分层）；`EXCLUDED` 57 列逐条带理由，`FEATURE_GROUPS` 51 个主模型特征
  - **新建** `src/floodrisk/models.py`：与算法无关的 `Bundle(backend, model, features, meta)` joblib 契约；`predict_proba` 强制 float64（XGB 默认 float32，写 CSV 只有 7 位有效数字会产生 1e-8 级假差异）
  - **新建** `scripts/run_small_experiment.py`：11 步验证（读表→目标→切分→锚定复测→分层抽样→特征→单变量 AUC→三模型→规则/强泄漏/特征族对照→位移敏感性→子进程独立加载）
  - **新建** `scripts/predict_from_bundle.py`：独立 CLI，只凭 bundle + 输入表出分，是第九阶段 `predict.py` 的最小形态
  - **新建** `tests/test_realdata.py`：12 项回归测试（目标不位移、跨网格不位移、不跳缺口、切分零重叠、特征/排除清单不相交、槽内未来列必在排除清单、规则基线复现标签、缺列报错…）
- 验证：`pytest` 160 项全部通过（原 148 + 12）。300k 抽样实验 30.5 s / 2.65 GB，7/7 验证通过，独立 CLI diff = 0.0。锚定复测：`prev6/12/24/48h` 对「截至槽开始」的 MAE 恒为 **0.000000**，对「含本槽」为 0.14–0.16 → **行的预测时刻就是槽开始，主目标无需位移**。对照实验：规则基线 AUC 1.0000（规则即标签生成器）；加入 `is_A/warn_level/is_event_day` 后 AUC 1.0000（证明排除清单必需）；`static_only`（零降雨）AUC **0.8963** vs 全特征 0.9238 vs `rain_time_only` 0.6631 → 判别力主要来自静态地形记忆。XGB test AUC 0.9238/PR-AUC 0.8460，RF 0.9377/0.8576（RF 四项指标优于 XGB）
- 遗留：**M1 静态记忆主导**（全特征只比纯静态高 0.028 AUC）在现有数据下无法靠换模型解决，只能等 A 补非事件日与空间覆盖；D1 没有任何一天覆盖全部 5,517 网格，全港地图需另构造推理输入；`val=2022` 是本研究的二次切分（A 的 `split` 只分 train/test/golden_eval），待用户确认；全量训练暂缓，等 `reports/04_small_experiment.md` §6 三个决策

### #6 · 2026-10-24 · v10.1 第二批交付核查：ERA5 补齐 + STPU 社区映射落地；锚定复测维持「槽开始」
- 意图：A 同学第二批数据（zip SHA256 7d643362…）到达，用户要求优先核查首轮四类问题（标签/降雨/全网格覆盖/社区映射）是否解决，不覆盖已有模型与结果
- 改动：
  - **新建** `data/delivery_v10_1/`（与 `data/delivery_v10/` 并存）：主表 v10.1（SHA256 9f580846…）+ 5 个 QC JSON + `grid_500m_static.csv`（5,517×66）+ `grid_500m_to_stpu.csv` / `grid_500m_stpu_overlap.csv`（5,517→211 STPU，含人口/65+/收入/脆弱性/暴露人口）+ `golden_grid_map_500m.csv`（87 行）+ 3 份文档（黄金事件清单/API 适配清单/字段符合性评估）
  - **新建** `scripts/check_delivery_v10_1.py`（只读核查，输出 `outputs/v10_1_check/summary.json`）
  - **新建** `reports/v10_1_batch2_check.md`
  - 未改动任何模型、bundle、`outputs/small_experiment_300k_20261024/`
- 验证（`check_delivery_v10_1.py` 实测）：v10.0→v10.1 行/列/标签/划分/潮位/POWER **完全不变**（7,200 行 × 7 列逐值对比 same_rate=1.0；决定性 0.999984、正例 4,223,633 不变），唯一实质变化 = **ERA5 9 列 NaN 47.3%→0%**（2018–2025 全 0，2026 余 0.04–0.05%）；ERA5 有空间信号（事件日槽内中位 3.5 个不同取值 vs POWER 恒 1）；锚定复测 POWER MAE_start≈3e-9 vs MAE_end≈0.2、ERA5 5e-8 vs 0.004–0.006 → **「截至槽开始」在 v10.1 继续成立，v10.1 README §8.0「截至槽末」表述与数据矛盾**；静态主表含全部 41 个静态模型特征 → 全港推理输入可构造；overlap 长表 211/211 STPU、每格权重和≈1（个别格为 0 需防除零）
- 遗留：是否用 v10.1 + ERA5 重跑 300k 实验、ERA5 进特征清单的方式（新增并列 vs coalesce），等用户对 `reports/v10_1_batch2_check.md` §六 决策 1/2 确认；非事件日覆盖（D2）与标签体系未变，M1–M6 其余限制继续有效

### #7 · 2026-10-24 · 方案 A 落地：ERA5 8 列入模 + 四组特征 × 三模型消融（v10.1）
- 意图：用户批准方案 A（ERA5 与 POWER 并列、保留来源标识）并要求同数据/同样本/同划分/同种子下比较四组特征方案，检验 ERA5 是否真正改善降雨驱动预测而非仅拟合弱监督标签
- 改动：
  - `realdata.py`：`FEATURE_GROUPS` 新增「降雨·前期累计（ERA5，截至 t）」8 列；`POWER_ONLY_FEATURES`（51 列）固化为 v10.0 快照；新增 `RAIN_FEATURES`（16 列）/`TIME_FEATURES`（3 列）修正口径——v10.0 实验用 `startswith("power_rain")` 划族曾把 `power_api72_mm` 误归静态族；EXCLUDED 移除 era5 prev/api72 8 列（v10.0 排除理由「训练期全缺」已失效）
  - **新建** `scripts/run_small_experiment_v10_1.py`：四组（power_only 51 / dual 59 / static_only 40 / rain_time_only 19）× 三模型 + rain_time_power(11) 诊断组；含行序一致性校验、power_only 对 v10.0 基线的逐位复现校验、独立加载 diff 校验、v10.0/v10.1 自动对比段
  - **新建** `reports/04b_v10_1_era5_ablation.md`；v10.0 全部产物与 `data_request_A.md` 保留不动
- 验证：`pytest` 160 项全部通过。消融实验 44.0 s / 13 模型：**power_only 在 v10.1 上三模型指标与 v10.0 逐位一致**（XGB 0.923838/0.846001）→ 对照前提硬验证成立。dual vs power_only：XGB AUC +0.0092（0.9238→0.9331）、PR-AUC +0.0265（0.8460→0.8725）、Brier 0.1094→0.1022；RF AUC 0.9451 / PR-AUC 0.8910（四项领先 XGB）；LR +0.0013（共线预期）。降雨族自身（无静态）：11 列 0.6608 → 19 列 0.6803，证明 ERA5 增益来自降雨信号本身。static_only(40) 0.8654 vs rain_time_only(19) 0.6803 → 静态主导结构未变（dual XGB top15：静态 11 席、ERA5 3 席、POWER 1 席，elevation 0.1022 居首）。独立加载 inproc/CLI diff 均 0.0
- 遗留：**16 列再分析降雨（POWER/ERA5 prev 族）部署期 t 时刻不可得**（发布延迟数天），第九阶段须以 HKO 实时源替代并做训练-部署 domain shift 设计；「ERA5 增益 = 信息增强 + 拟合弱监督规则增强」的混合在当前标签体系下不可完全分离，已写入论文限制；正式训练待用户确认 dual 方案后另行批准

### #8 · 2026-10-25 · 第四章描述性统计 + warn_level 与降雨脱钩的重大口径发现
- 意图：论文第四章「历史记录的时间与成因特征」需要可复现、口径严谨的描述性统计；真实事件层与弱监督层分开、不训练不修改任何数据/模型
- 改动：
  - **新建** `scripts/chapter4_descriptive_stats.py`（只读：主 Parquet 轻列全表 + filter_keys 两遍读；黄金事件清单转录为脚本常量；`slots_covered()` 警告时段→2h 槽映射）
  - **新建** `reports/chapter4_descriptive_statistics.md` + `outputs/chapter4_statistics/`（table_summary.csv 23 行 + 分组 CSV 12 个 + _meta.json）
  - 未改动主 Parquet、划分、任何模型产物
- 验证：脚本两次跑通，输出自洽。关键结果：真实事件 6（窗口内 2，28 地点→87 对→网格并集 75→STPU 35/59）；警告活跃槽 459 行 ERA5 prev6h 中位 5.49mm vs 非事件日 p95 6.72mm；天文潮位事件窗口不偏高（中位 1.33m）；复合条件强降雨 30.3%、高潮位共现 0%（增水不在表中）。**重大发现**：376 个事件日中 306 天（81%）全港 POWER 12 槽合计中位日雨量 <1mm 却 warn_level≥2（抽查 2021-11-11/2024-12-16/2025-01-15/2025-01-28 全≈0mm），≥100mm 仅 1 天；黄金事件日 warn_level 与降雨一致 → 弱监督正例月份分布（1 月 22.61% 最高、冬季 58.4%）与香港暴雨季节相反，不可解读为真实季节特征
- 遗留：warn_level 口径须 A 澄清（解析/对齐错误还是定义含非降雨警告；是否需重新生成 is_event_day/flooded_2h）——已写入报告 §7 注，未覆盖 data_request_A.md；此前弱监督指标的解释均需保留此限制

### #9 · 2026-10-25 · 第四章四张表可填数据提取 + STPU 属性三项核查
- 意图：论文第四章四张表格（历史记录/验证设计/社区脆弱性/黑点治理）需要「当前能确认什么、什么必须等待」的口径纪律化提取；不训练、不改数据
- 改动：
  - **新建** `scripts/chapter4_extract_existing_data.py`：只读复用黄金清单/gmap/to_stpu/overlap + 上一脚本产物 + metrics.json，不重读主 Parquet
  - **新建** `reports/chapter4_tables_available_now.md` + `reports/chapter4_remaining_data_gaps.md`（A/B/C 三类缺口）+ `outputs/chapter4_tables/`（14 个 CSV，含正式验证表空白结构）
  - 未改动主 Parquet、划分、任何模型产物
- 验证：脚本一次跑通。关键核查：①STPU 七个属性跨格零不一致（确为 STPU 级常量，可按 STPU 汇总不重分配）；②pop_total 211 个加总=7,380,883 恰等 2021 普查总人口→普查口径确认；③exposure_population 与 pop_total 逐值恒等→「潜在暴露人口」无独立口径，标待 A 澄清；④SVI 与 rank Spearman=−1.0→rank 1 最脆弱。表 2 核实：实验切分 train 2018–2021/val 2022/test **2023-01–2026-09**（论文若写 2024-01 起须修正）；空间分组验证未运行→待正式测试。表 4：无 DSD 黑点名单数据，按「已核实历史事件地点表」口径交付 29 行
- 遗留：A 澄清清单 6 项（warn_level 口径/SVI 定义/exposure 定义/人口年份/dc 对照/黑点名单）；B1–B7 待正式训练与用户批准

### #10 · 2026-10-25 · 训练前最终审计 + 数据版本冻结 + 事件日-降雨脱钩 P0 发现
- 意图：最后一批交付（delivery_v10_20261025.zip）到达后，按任务书完成正式训练前的数据版本审计、逐特征时间泄漏审计、标签/抽样偏差审计与正式训练方案设计；不训练、不改数据
- 改动：
  - **新建** `scripts/final_pretraining_audit.py`（只读审计：三版本哈希/行列/标签分布、全表规则验证、抽样结构、黄金独立性、逐年 NaN、诊断集 KS 对比）→ `outputs/final_audit_20261025/`（6 个 JSON/CSV）
  - **新建** 5 份报告：`reports/final_data_inventory.md`、`pretraining_leakage_audit.md`、`sampling_and_label_audit.md`、`formal_training_plan.md`、`decisions_for_user.md`
  - 新交付解压至 `data/delivery_v10_20261025/`（19 文件，还原 GBK 文件名），未覆盖任何历史目录
- 验证：`pytest` 160 项通过。关键实测：**主 Parquet 与 v10.1 SHA256 逐字节一致**（9f580846…dea7822），正式版本冻结于此哈希；新增仅诊断集 3 文件 + 质询回复；`(grid_id,date,slot)` 重复 0 行；标签规则覆盖 99.995% 正例（仅 193 黄金修正）；黄金 (grid,date) 与 train/test 零重叠但 75/75 网格在 train 有其他日期行、2018-09-16 有 28,452 行同日 train 样本；诊断集 label_verified 恒 0、ERA5 缺测 0、抽样偏差集中于静态特征（elevation/impervious KS 4/4 显著、POWER 降雨格间零方差）。**P0 发现：376 个事件日中 219 天（58.2%）全港 POWER 雨量精确为 0、306 天（81.4%）格均<1mm，事件日格均雨量中位数（0.0mm）低于普通日（1.18mm），无雨事件日成串出现在 1-2 月，真实暴雨样子集仅 48 天（≈5.5 天/年，与气候学吻合）→ warn 解析真实性待数据侧定点核查（D1）**
- 遗留：D1–D4 四项 P0 决策待用户批复；D1 澄清前不启动全量训练

### #11 · 2026-10-26 · D1 警告日期真实性核查：根因锁定（火灾警告混入），95.3% 正例受影响
- 意图：用户批准暂缓训练并要求用 HKO 官方暴雨警告数据库核查 376 个事件日真实性；区分解析错误 vs POWER 低估；只核查不修数据
- 改动：
  - **新建** `reports/warning_date_authenticity_audit.md`（全量对照 + 根因鉴定 + 修复方案）
  - **新建** `outputs/warning_audit_20261026/`：event_days_376.csv、sample_dates_to_verify.csv、hko_rstorm_official.csv、hko_rstorm.dat.raw / hko_fire.dat.raw（官方原始文件留档 + SOURCE_HASHES）
  - 数据源：HKO 官方 warndb 底层文件 rstorm.dat（975 条分钟级记录，1998 起）+ fire.dat（1,429 条，根因鉴定用），2026-10-26 抓取
- 验证（全量 376 天逐日对照官方库，非抽样）：官方红/黑覆盖日仅 **56 天**，我方 56/56 全命中且等级零错配、**零漏报**；假日 **320 天（85.1%）**官方无任何级别暴雨警告。根因：HKO 月度摘要警告表中"红色火灾危险警告"被并入"红色暴雨警告"——窗口内官方红色火灾日 316 天与我方假日 1:1 完全重合零漏检，另 2 天为黄火日，2 天（2021-12-30/31）根因待 parsed_warnings.json；槽级时段与官方火灾/暴雨时段双向精确对齐（如 2025-01-03 红火 06:00-23:15→槽3-11；2018-09-16 红雨 10:55-18:50→槽5-9）。**影响：4,026,940/4,223,633（95.3%）正例落在假事件日；修复后数据量约 290 万行、正例率约 6-7%**。双源（POWER+ERA5 均为 0）+官方无记录+旱季聚集+气候学频次四重证据排除 POWER 低估
- 遗留：修复方案 R1（数据侧重生成 warn/event_day/flooded 列 + sample_frame，特征列不动，约 1-2 工作日）/R2（需单独批准的我方重建）待用户批准；2021-12-30/31 两天根因待 A 方 parsed_warnings.json；D2-D4 已确认（消融待标签修复后执行、不预设双主模型、接受潮汐全期拟合并披露）；暂停等待批准，未动任何数据

### #12 · 2026-10-26 · B 侧 R2 修复落地：官方警告源重建 delivery_repaired_b1 + 轻量实验确认降雨信号真实
- 意图：用户批准 R2 方案（A 方退出，B 侧独立修复）——以 HKO 官方 rstorm.dat 为唯一警告源重建标签与抽样框架，修复 95.3% 正例被火灾警告污染的问题；本轮只做修复 + 验证 + 一次 ≤30 万样本轻量实验，全量训练仍待批准
- 改动：
  - **新建** `scripts/b_side_repair.py`（feasibility/repair/validate 三子命令，seed=42）：位置化流式替换 10 个标签列（warn_level~sample_role），320 个假事件日改判非事件日并确定性重抽 35 格/日；59 特征与其余 89 列逐行不动；黄金 936 行标签逐值保留
  - **新数据版本** `data/delivery_repaired_b1/`（主表 + sample_frame + manifest_repair_b1.json，全部 SHA256/种子/参数/差异统计入 manifest）；旧版本零改动，诊断集只读隔离
  - **新建** `scripts/run_small_experiment_repaired_b1.py`（与 v10.1 对照实验同设计：5 特征组 × XGB/RF/LR + 规则基线 + bundle 复现）→ `outputs/small_experiment_repaired_b1_300k_20261026-013127/`
  - **新建** 4 份报告：`reports/b_side_repair_feasibility.md`（9 项门槛 GO，发现峰值规则依赖未交付 ERA5 中间量→改 keep_old_on_correct_days 策略）、`b_side_repair_validation.md`、`b_side_old_new_comparison.md`、`b_side_next_training_decisions.md`
  - **新建** `CHANGELOG.md`（首条 = 本版本）
- 验证：`pytest` 167 项通过；修复验证 11/11（主键 0 重复、56 事件日=官方、354,441 红黑槽行 0 错、正例来源违例 0、黄金逐值保留、59 特征抽查逐值相等、划分互斥、诊断集 SHA256 不变）。新规模 2,905,896 行 / 正例 196,693 / 正例率 6.7688%（官方推演结果，非调节目标）。轻量实验（300k，test 正例率 8.45%）：dual_xgb ROC 0.757 / PR 0.189 / Brier 0.0795；静态主导下降雨贡献显著上升（重要性 top15 降雨族 11 席），ERA5 增益比 v10.1 更明确（dual−power_only ROC +0.032）；泄漏体检 0 可疑；跨版本指标不可直接比较（PR-lift 2.23 vs 2.43 同量级）已声明
- 遗留：正式全量训练待用户批准；训练前需决策 T2-T6（特征主方案、val 2022 仅 2 红黑日 789 正例的处理、0.3 阈值不适配新先验需重做阈值-成本与校准、cv_fold 需重检、320 改判日 A 类占比 55.5% 受限重抽偏差披露）；2021-12-30/31 两日根因仍待 A 方 parsed_warnings.json（不影响本轮修复，该两日按官方无警告处理）

### #13 · 2026-10-26 · 正式全量训练（delivery_repaired_b1）：dual XGB 主模型 + LR 基线 + 消融 + SHAP/GIS，量化确认静态记忆主导
- 意图：用户批准 9 项条件下启动正式训练——仅用修复版数据、dual XGB 主模型 + power_only 对照 + LR 基线、test 只做最终评估、val 上重选阈值、输出全套指标/SHAP/GIS，并如实披露偏差
- 改动：
  - **新建** `scripts/run_formal_training_repaired_b1.py`（--preflight 预检 + 正式训练一体）：SHA256 冻结、时间锚/主键/划分零重叠/特征注册表/训练段单变量 AUC 六道中止闸门；5 模型（dual_xgb / dual_lr / power_only_xgb / static_only_xgb / rain_time_only_xgb）无调参；阈值每模型在 val 上 0.01–0.99 扫描 F1 最大（代码路径分离保证 test 不参与选阈）；SHAP 用 xgboost pred_contribs 精确 TreeSHAP；GIS 逐格聚合输出 QGIS 可联接 CSV
  - **新产物目录** `outputs/formal_training_repaired_b1_20261026-020814/`（metrics.json、5 bundles、6 图件、predictions_val/test.parquet、5 份阈值扫描表、gis/grid_risk_500m.csv 5517 格）+ `reports/formal_training_results_repaired_b1.md`
  - 环境：uv 安装 matplotlib 3.11.2（SHAP 用 xgboost 原生 pred_contribs，未引入 shap 包）
- 验证：预检 6 门槛全过（9.2s）；pytest 167 项通过。**test(2023–2026.09, 1,496,340 行, 正例率 8.49%) 主结果**：dual_xgb ROC 0.751 / PR 0.184 / F1 0.241 @ τ=0.05（val-F1 最大点，P 0.192 / R 0.324）；dual_lr ROC 0.750 / **PR 0.291** / F1 0.293 @ τ=0.89（时间外更稳健）；power_only_xgb ROC 0.724；static_only_xgb ROC 0.720；rain_time_only_xgb ROC 0.647。ERA5 增益 +0.027 ROC。golden 敏感性 ROC 0.904（加密样本仅供参考）
- 关键发现（写入报告 §5.2）：**① val(2022，仅 2 个红黑日) ROC 0.961 → test 0.751 泛化差显著，论文引用以 test 为准；② 静态地形记忆主导判别力（静态 40 列单独 0.720，动态降雨边际 +0.031），证实此前疑虑——当前模型是「已知地形下的风险空间分辨」而非降雨驱动的 2 小时动态预测；③ 事件日内部空间分辨中等（event_day_1 内 ROC 0.638）；④ 旧 0.3 阈值在修复版上 F1 仅 0.014，重选 τ=0.05 必要**
- 遗留：18 折完整空间 CV 未执行（用户允许）；cv_fold 实为 {-1,0..17}（-1 为 16,212 行附加组，网格→折唯一成立）；「去潮位」消融不在本轮批准范围未训练；数据 SHA256 `1749cc00…5cb4e8` 未变，无任何旧产物被覆盖

### #14 · 2026-10-26 · 最小补充实验：去潮位消融 + 一致性 + 动态能力 + 阈值敏感性 + 雨潮复合（用户 5 项批准）
- 意图：正式训练后用户批准的最小补充实验（不做 18 折 CV / 大规模调参 / 概率校准），验证论文「雨潮复合」论断与模型泛化能力，并为论文三、四章提供行号级修改依据
- 改动：
  - **新建** `scripts/run_supplement_tide_ablation_b1.py`（step0 冻结哈希+潮位特征核查 → step8 落盘，八步）：no_tide_xgb 仅移除潮汐组 4 列（`tide_pred_max_m / tide_pred_mean_m / tide_n_hours / dist_to_tide_m`），其余与 dual_xgb 完全一致（seed=42、M.build_model、val F1 选阈）；bundle 重预测一致性核对；group_metrics（全部样本/事件日/分年 + 沿海/内陆）；aggregated_auc（日期级/grid-day 级）；test 阈值敏感性（仅敏感性分析，不重选阈）；rain_tide_composite（雨三分位×潮三分位，含 mean_p + obs_rate 实际正例率，沿海/内陆分层）
  - **新产物目录** `outputs/supplement_tide_ablation_b1_20261026-024612/`（metrics.json、results_report.md、no_tide bundle、predictions_test_no_tide.parquet、PR-AUC 诊断表、校准十分位表、沿海分组表、雨潮复合表×2、阈值敏感性表×2）+ `reports/supplement_tide_ablation_b1.md` + `reports/paper_updates_ch3_ch4_b1.md`（论文三/四章行号级修改清单）
- 验证：pytest 167 项通过。**test 关键结果**：no_tide_xgb ROC 0.7579（**Δ vs dual_xgb = +0.0070**）、PR 0.1929（Δ+0.0093）、Brier −0.0009 → 判别力不由潮位贡献，移除潮位反而略升；bundle 重预测概率逐值 diff=0.0e+00（XGB/LR 均是），y 逐行一致；动态能力分解 dual_xgb 全样本 ROC 0.751 / 事件日内 0.638 / **日期级 0.897** / grid-day 级 0.950 → 模型强在「哪天有风险」与静态空间分辨，弱在事件日内小时级分辨；沿海组 ROC 0.576 显著弱于内陆 0.802；test 阈值敏感性：旧 0.30 阈值 Recall 仅 0.072，维持 val 选定 τ=0.05
- 关键发现（写入报告 §7）：**雨潮复合「标签层与模型层方向相反」——事件日标签层「雨高×潮高」实际正例率 22.90%（vs 其余 12.78%，+10.12pp；沿海达 41.88%），但模型预测概率反而更低（0.0281 vs 0.0499，−0.0217）→ 弱标签存在雨潮复合正例聚集，模型未学到该交互；论文只能表述为「假设与探索方向」，不可宣称复合预警能力**
- 遗留：LR PR-AUC 优于 XGB（0.291 vs 0.184）的机理已定位（顶端排序更尖 precision@1000 0.776 vs 0.407 + balanced 权重概率尺度失真）但未做任何重调参（遵守约束）；未启动任何新训练任务

### #15 · 2026-10-26 · 最终核验 + 空间应用分析：聚合 AUC 口径审计、雨潮组成效应修正、空间排名对比、GIS 口径证实、社区排名拒生成
- 意图：用户批准的最后一轮只读核验（不训练/不改数据/不改旧报告），为论文收尾提供"能写什么、不能写什么"的边界清单
- 改动：
  - **新建** `scripts/run_final_verification_spatial_b1.py`（step0 SHA 冻结 → 四项核验 → 落盘，全程只读）：①日期级/格日级聚合 AUC 复算（聚合函数、标签定义、单元数、同日计权、日内信息回流审查）；②雨潮 27 组合复算（新增逐组合正例数与 mean_is_A/mean_low_coastal 组成检验）；③dual_xgb vs no_tide_xgb 逐格排名（主口径=test 期格内均值、敏感性=格内最大值，Top20+Spearman）；④人口/脆弱性/社区边界字段可用性盘点
  - **新产物** `outputs/final_verification_spatial_b1_20261026-030632/` + `reports/final_verification_spatial_b1.md`（三层结论清单：直接支持/谨慎解释/无法证明）
- 验证：pytest 167 项通过；全部复算与存储值一致（聚合 AUC diff=0.00e+00，雨潮复合 n diff=0、mean_p diff≤4.9e-07）
- 关键发现：
  - **日期级 0.897 / 格日级 0.950 不能支持"提前 2 小时动态预警"**：p_max 聚合含事件后槽位（日内信息回流），标签为事后确定的日级警告属性；符合 2h 口径的仅事件日内行级 ROC 0.638；格日级 0.950 系 12 槽任一正例的容易任务（基线 34.7%）
  - **雨潮"组成效应"假设修正**：各雨/潮箱 mean_is_A 几乎恒定（0.552–0.557），"雨高×潮高 22.90%"抬升并非 is_A 组成机械造成；但组合正例率雨维**非单调**（雨高×潮中最低 6.74%）且模型评分最高组合（雨中×潮中 0.103）恰为正例率最低之一——标签层聚集受警告归因窗口槽级时序结构影响，模式与物理交互预期不符，仍只能作假设表述
  - **GIS 口径证实**：`mean_risk_all` 为全期 train+val+test 非 golden 逐格平均（行数复核 5,517/5,517），含训练期数据，非 test 时段风险；论文引用必须标注口径
  - **空间排名**：Spearman 0.9888（格局稳健）但 Top20 重合仅 9/20（头部概率差 <0.008，名次对特征集敏感）
  - **社区排名拒生成**：无社区/街道边界映射字段；人口/脆弱性字段为格级常量且按设计未进模型——任何社区级排名都依赖虚构映射
- 遗留：格级"风险×人口"后验叠加展示未生成（需用户单独批准）；test-only 空间排名已产出（top20_grid_rank_comparison.csv），可作为 GIS 全期口径的替代

### #16 · 2026-10-26 · 论文成果整理包：6 模型最终对比表 + 论文图件 + 三/四章素材包 + 全文逐段审计（只读）
- 意图：用户批准的论文整理阶段（停止训练/不训练、不改数据、不改论文原文），把全部既有实验结果整理为论文可直接引用的图表、素材与逐段修改依据
- 改动：
  - **新建** `scripts/run_paper_package_b1.py`（只读整理）：SHA 冻结 → 6 模型对比表（数值直取 formal/supplement 冻结 metrics.json 不重算，no_tide 取 step3_ablation）→ 测试期逐格风险表（dual/no_tide 均值+最大值+排名+Top20 标志，联接 GIS 全期均值仅作参考）→ 5 张论文图（ROC/PR 六模型、雨×潮热力图 3 口径×2 层、测试期 vs 全期网格地图并排、排名稳定性散点、SHAP Top20）→ 汇集既有 CSV 与 6 张正式原图至 `formal_originals/`
  - **新产物目录** `outputs/paper_package_b1_20261026-122643/`（figures/ 5 张新图、tables/ 6 份 CSV 含 `grid_risk_test_period_500m.csv`、formal_originals/ 6 张、metrics.json）
  - **新建** `reports/paper_material_ch3_ch4_b1.md`（三/四章完整素材包：方法/数据来源/弱监督标签/样本划分/指标口径分层/局限 + 图注建议 + 三层引用速查）
  - **新建** `reports/paper_paragraph_audit_b1.md`（后台代理起草、本人逐数核定：27 条不一致 + 16 条需加限定 + 保留确认；最严重 5 条 = L88 标签口径、L402 结论导语 RF/留一/社区尺度三重不实、L403 雨潮写成已验证结论、L404 社区级产出虚构、L408 三项披露缺失）
  - **修订** `reports/paper_updates_ch3_ch4_b1.md` §1：加更正标注——旧稿分年 ROC（0.7538/0.8201/0.7719/0.7427）为誊录错误，正确值为 metrics.json by_year 0.7442/0.8311/0.7752/0.7220
- 验证：pytest 167 项通过；对比表数值与冻结 metrics.json 逐一核对一致；Spearman 0.988753 / Top20 9/20 与最终核验报告完全一致；审计报告关键数字抽查通过（val@0.05 P 0.3554/R 0.8223/F1 0.4963；val@0.30 F1 0.0142；tide_pred_max_m SHAP 第 11=0.2493；全期 top1 H078033=0.1149 vs 测试期 top1 H092081=0.0990）；SHAP 图标题错误（"潮位未进前 20"）在核查中发现并修正为"列第 11、非主导"
- 关键发现：**GIS 全期与测试期口径差异实证**——全期 top1 网格 H078033 在测试期仅列 21，测试期 top1 H092081 全期列 29，两口径排名不可混用；论文实证展示优先用测试期
- 遗留：2 条证据不足项（L86 清洗校核记录、L145 独居长者/低收入字段）待项目负责人裁决；论文原文未做任何修改

### #17 · 2026-10-26 · 智慧防洪 MVP：HKO 实时接入 + GIS 历史评分 + 整合 Dashboard（与论文实验数据完全隔离）
- 意图：论文/比赛演示所需的可运行 MVP——实时 HKO 数据接入、5,517 网格历史模型评分地图、三类信息（观测/官方预报/历史评分）明确区分的整合展示；全程不触碰已冻结数据集、模型、测试集预测、SHAP 与 GIS 历史评分，不展示未经验证的未来 2h 水浸概率
- 改动：
  - **新建独立目录 `hko_api/`**：`hko_fetch.py`（5 源轮询抓取：warnings/rhrread/hourlyRainfall/hhot/nowcast，纯标准库；原始响应按日期归档 + .meta.json + fetch_log；每源 obs_time 与 fetch_time 双时间戳；失败保留上次成功快照并标 stale）+ `prepare_scores.py`（只读 formal_b1 测试期预测 → 5,517 格测试期/全期双口径评分，中心点呈现不伪装边界）+ `dashboard.html`（本地 Leaflet 离线可用；双评分图层、nowcast 预报叠加开关、STALE 横条、逐源新鲜度表、弹窗含弱监督披露）
  - **关键修复（影响行为正确性）**：`build_snapshot` 原 `fetch_time = entry.get("fetch_time") or hkt_str()` 导致成功抓取后 fetch_time 永不更新 → 客户端 STALE 常驻误报；改为成功写本次获取时刻、失败保留最后成功时刻；同批修复 `run_once(only)` 增量合并覆盖整源、HHOT obs_time 误取未来时刻、图例拦截点击（pointer-events:none）
- 验证：pytest 167 项通过（主套件无回归）；E2E 8 项全 PASS——5 源实时抓取成功、5517 格渲染、popup 点击经 canvas 真实代码路径验证（H078033 测试期 0.0915/全期 0.1149+披露）、死代理断网演练（stale 标记+错误记录+fetch_time 保留+横条常驻）、部分降级混合态、恢复全 OK、obs/fetch 时间语义分离、file:// 打包模式；汇报见 `hko_api/reports/mvp_readout.md`，API 核验见 `hko_api/reports/hko_api_inventory_and_feature_mapping.md`（修正 A 方清单 3 处口径）
- 遗留：prev24h/48h/api72 特征需 72h 轮询积累（动态评分核心缺口）；P1 网格-观测联动与演示模式未做；未来接入 XGBoost 实时推理须独立版本+特征兼容性验证+用户批准，不得自动启动

### #18 · 2026-10-26 · 论文四表实证填充：证据核对 → 填表 → 修订版 → Word审核副本 → 证据追溯报告（全程只读，不训练）
- 意图：用户指令进入《智慧防洪(4).docx》表格实证填充阶段，依据已冻结的 delivery_repaired_b1、6 模型正式训练结果、GIS 产物与验证报告填写四张目标表（表①历史事件统计、表②验证设计与指标、表③社区清单、表④黑点清单）；严格区分真实历史事件 / 弱监督标签 / 模型评分 / GIS 空间属性，不依赖实时 API
- 改动（全部在新独立目录 `outputs/paper_tables_fill_20261026/`，未覆盖任何旧产物）：
  - `01_四表核对清单.md`（字段—来源—证据—状态—建议）、`02_四表填写与修订版.md`（有证据填格 + 表②修订结构 + 表③/④替代附表方案）
  - 只读联表 2 份：`app_places_with_test_risk.csv`（28 报道地点 × 测试期匹配格评分：前 3 = 沙头角 0.0869/排名 95、大埔 0.0866/99、海怡半岛 0.0856/132，与测试期 Top20 无一重合）、`app_grid_candidate_checklist.csv`（测试期 Top20 网格 × 人口/65+/收入，H084005/H073015 缺失已标注，Top20 多为 WorldPop<5 人低人口格）
  - `智慧防洪(4)_表格审核副本_20261026.docx`：unpack→XML 编辑→pack 生成；表① 5 格、表② 填 6 实训模型（各模型自选 val 阈值 τ=0.05/0.89/0.05/0.07/0.03/0.04，克隆插 4 条消融行）、RF 行标"未训练"、空间分组标"未执行（仅完成有效性检查）"、表③ 不生成社区排名占位+附表 A/B 说明、表④ 以"报道地点候选核查清单"口径填 3 地点；插副本说明、4.2.1/4.4.1 修订版方法段、表② 表注 6 条、表④ 口径声明 5 条；原文档 SHA 未变
  - `03_表格证据追溯报告.md`：逐项注明数据文件、字段、口径、时间范围、计算方式与缺失限制
- 验证：本次为纯文档/证据核对工作，无代码改动（pytest 基线不受影响）；pack.py OOXML 校验全部通过（Paragraphs 451→492）；结构复核 `<w:tbl>` 9→9、`<w:tc>` 245→269、`<w:tr>` 57→61、sectPr/body 完整；打包后逐格复读四表确认填充到位；表② 全部数值与 metrics.json/final_model_comparison.csv 逐一核对一致
- 遗留：审核副本中长文本在固定列宽下排版需人工在 Word 复核；SVI 定义、warn_level 口径两处仍待数据方澄清；渠务署黑点名单、社区边界映射缺位决定表③/④ 只能以候选核查清单与占位方式交付

### #19 · 2026-10-26 · 表③v2：按论文§4.3双阈值规则填写STPU级高风险高脆弱候选清单（只读计算，不训练）
- 意图：承接 STPU 映射只读核查结论（映射真实可核验、与测试期评分 100% 可关联）与用户指示"根据这个图表填写"，将论文表③（高风险高脆弱社区清单）由 v1 占位升级为 STPU 级候选核查清单
- 改动（全部在 `outputs/paper_tables_fill_20261026/`，未覆盖任何旧产物、未修改原始数据）：
  - **新建** `app_stpu_candidate_checklist.csv`（207 个有属性 STPU 全量底表）：按 A 方 README §3.1 自订面积加权公式（Σ(格评分×overlap_frac)/Σoverlap_frac）对冻结测试期格级评分（grid_risk_test_period_500m.csv）做 STPU 聚合，联接 2021 普查人口/65+/收入/SVI 并标注五分位与双阈值标志，6 个候选置顶
  - **更新** `智慧防洪(4)_表格审核副本_20261026.docx`（v2 重打包）：表③ 21 格全部填写（评分降序前 3 候选 STPU 115/217/211，含口径限定）＋表后说明段替换为含 5 条限制的 v2 说明；原版 docx SHA `540fdd9524bd776f` 未变
  - **更新** `02_四表填写与修订版.md`（表③节 v2 改写＋附表 C）、`03_表格证据追溯报告.md`（新增 E9 证据行＋第 8 节 v2 补充追溯＋全局限制 4 修订为"STPU 映射已核验、治理社区映射仍缺位"）
- 验证：重新 pack OOXML 校验全部通过（Paragraphs 451→492，v2 为原位替换未增段落）；打包后复核表③ 4 行 28 格全部为 v2 内容、"本轮不生成社区排名"出现 0 次、表①②④填充与 9 个顶层表结构未受影响；6 候选（115/217/211/976/112/944）为评分前 10%（21/207）∩SVI 前四分位（52/207）交集，数值与 CSV 逐一核对一致
- 关键口径：候选核查序≠治理优先级（综合公式未合成、未获批）；人口为 2021 普查常住口径非受灾人数（exposure_population≡pop_total 未采用）；评分前 3 候选人口偏少（1,908–13,664 人），"评分高≠受影响人口多"必须并列表述；治理社区（区议会选区）级排名仍不生成；SVI 计算方法待 A 方补充
- 遗留：候选 STPU 中文名称需与 A 方 STPU 表或普查公开表核对后补充；桌面交付副本同步更新

### #20 · 2026-10-27 · T3：STPU 区域级底表 + 映射/权重/聚合/人口全面核验（只读输入，不训练）
- 意图：把「5,517 网格测试期评分 → STPU 统计小区」的聚合链路做成可复现、可审计的底表，先解决 207 vs 211、面积权重方向、人口重复计数、SVI 可复算性四个疑点，再谈区域分析
- 改动（全部独立输出，未覆盖任何旧产物）：
  - **新建** `scripts/build_stpu_base.py`：6 项内置断言（网格集合一致、主键唯一、属性为 STPU 常量、风险无缺失、权重长表无重复、聚合可复现）；只读 5 个输入（映射表、面积权重长表、映射 QC、冻结测试期网格评分、网格静态表）
  - **新建** `outputs/t3_stpu_base_20261027/`：`stpu_base_table.csv`（211×20）、`field_dictionary.csv`、`data_source_table.csv`（含 SHA256）、`quality_report.md`、`open_issues.md`（13 项）、`acceptance_report.md`、`qc_summary.json`
- 验证：
  - 207 vs 211 定位：差异 4 个 STPU（122/214/228/231）各仅 4 个重叠网格、最大重叠 0.1536–0.3543，从未成为主映射；长表仍覆盖 → 两套口径不矛盾
  - 权重方向核验：3,615 格权重和≈1、1,900 格 <1（含 4 格为 0）、0 格 >1；主映射表权重仅 3 位小数（最大差 5e-4、20 行被舍入为 0）→ 聚合改用长表 4 位权重
  - 聚合复现：独立按 Σf·p/Σf 重算 207 个 STPU，与既有 `app_stpu_candidate_checklist.csv` 的 `stpu_score_test` 最大差 **4.95e-07**、网格数 207/207 一致、权重和差 2.8e-14；面积加权 vs 简单平均相关 0.9034、最大差 0.0261（权重不可省略）
  - 人口：STPU 级常量经逐组 nunique 验证；207 个合计 7,380,883 = 2021 普查合计；`exposure_population ≡ pop_total` 已识别为重复计数陷阱并标不可用；WorldPop 交叉 r=0.929
  - SVI：无法复算（范围 −1.84–5.93，排名含并列 206/207），按用户要求保留字段并标「待验证」，不参与任何筛选或排序
  - `pytest` 167 项通过（未改核心代码）
- 遗留：P0 三项阻断正式区域分析——STPU 边界缺失、2021 普查 STPU 底表未交付（`census2021_stpu_units.csv` 不存在）、SVI 方法未公开；故本轮不生成高风险×高脆弱交集、不生成区域排名与地图

### #21 · 2026-10-27 · T6：香港 18 区气象播报页（官方观测 + 规则模板，不做实时推理）
- 意图：在产品轨道落地"用户任选一区、查看该区过去 1 小时降雨与全港警告"的最小可用功能，同时把实时气象与历史风险图层彻底分开，避免误读为实时水浸概率
- 改动（新增独立目录 `hko_api/district/`，未改动 `dashboard.html`、`hko_fetch.py`、`hko_api/data/`）：
  - **新建** `index.html`：18 区按钮选择器（localStorage 记忆）、播报卡片、5 行键值（雨量/窗口/obs_time/fetch_time/状态）、全港级警告卡片、数据新鲜度表、历史风险独立入口
  - **新建** `bulletin.js`：纯函数规则模板（雨量 6 级分档、全港警告翻译、缺测/过期/失败分支），不读时钟、不生成任何预报数值
  - **新建** `tests/bulletin.test.mjs`（8 组规则测试）、`reports/t6_field_mapping.md`、`reports/t6_acceptance.md`
- 验证：
  - 规则测试 `node tests/bulletin.test.mjs` → **8 passed / 0 failed**（含 10 个雨量边界、缺测不推算、接口失败不编数值、过期强制标注、多警告并列、输出不含预报/概率）
  - 生产页面 E2E：18 区齐全、默认「中西區 0 mm」、切换沙田/大埔标题与数值同步、警告卡片标「全港級資訊」、新鲜度 `warnings OK / rhrread OK`、风险入口 `../dashboard.html`
  - 首轮发现并修复：`renderBulletin` 更新标题时用 innerHTML 覆盖 `#bhead`，误删 `#blevel` 节点 → 后续调用 `Cannot set properties of null`，播报区停在占位文字；改为只更新 `#blevel` 的文本/类名，复测无本页 JS 异常
  - 异常态实测（`/tmp` 固定快照副本，不触碰生产归档）：过期态 → 橙条 + `⚠` 首段 + 徽标「過期快照」+ 两源 STALE；失败态 → 「無分區數據」+「氣象數據暫不可用」+ 两源「無數據」，全程无编造毫米数值
  - `pytest` 167 项通过（未改核心代码）
- 遗留：无 18 区边界几何（仅列表选择器，不做分区填色）；官方 nowcast 无分区产品（不展示分区未来 2h）；`dc` 编码无区名映射（不做网格风险按区聚合）；浏览器截图工具不可用（以 DOM 文本 + 控制台状态留证）

### #22 · 2026-10-27 · T3 研究质量审计：人口口径 / 权重几何 / 聚合 / SVI（只读，不训练）
- 意图：对上一轮 STPU 底表做独立质量审计，重点验证人口合计口径、4 个无属性 STPU 影响、面积权重几何含义、聚合公式与 SVI 可复算性；逐项给出「通过 / 待验证 / 不适合正式使用」
- 改动（全部独立输出，未改任何输入）：
  - **新建** `scripts/audit_t3_quality_20261027.py`：17 项自动核查（属性常量、人口合计、官方口径、权重和、主映射=最大重叠、面积反推、聚合复现、主映射表偏差、加权 vs 简单平均、SVI 回归、暴露人口、WorldPop 交叉、重复值、65+ 自洽、收入单位、风险完整性）
  - **新建** `outputs/t3_quality_audit_20261027/`：`research_quality_audit.md`、`audit_checks.csv`、`population_crosscheck.csv`、`audit_summary.json`
  - **更新** `reports/paper_delivery_checklist.md`：G5/G6/G9/G10/M9 补入本轮审计证据与更正
- 验证：
  - 17 项核查 = 通过 11 / 待验证 4 / 不适合正式使用 2
  - 人口：207 个合计 7,380,883 = A 方 QC 的 `census_total`，但官方 2021 普查为 7,413,070，差 32,187（0.43%）→ 不能认定「等于全港人口」；原始底表 `outputs/v7/census2021_stpu_units.csv` 未交付
  - **新增 P0 发现**：逐 STPU 与 WorldPop 比值 46/207 超出 [0.5,2]，范围 0.15–24.65（`121S` 普查 1,153 vs WorldPop 28,417；`245` 45,645 vs 6,716）；全港 r=0.929 掩盖局部键错配风险 → 人口侧暂不用于正式区域比较
  - 权重几何：精确求和 3,500 格=1、1,965 格 <1、52 格 >1（4 位小数舍入，最大 0.0002）、4 格=0；更正上一轮「0 格 >1」表述
  - 面积一致性：`pop_total/pop_density` 反推面积 1,111.0 km² ≈ 权重和×0.25 的 1,111.0 km²（逐区 0.997–1.003）→ 密度由同一几何派生、STPU 覆盖完整
  - 聚合：独立复现 Σf·p/Σf 最大差 4.95e-07；若误用 3 位主映射表，最大偏差 0.0316、排名变动 98 位 → 必须用 4 位长表
  - SVI：三字段回归 R²=0.647，不可复算；`exposure_population ≡ pop_total`（207/207）→ 两项均标「不适合正式使用」
- 遗留：P0 三项仍未解除（STPU 边界、普查原始底表、SVI 方法），新增第四项（人口键错配排查）；本轮未生成任何排名或复合指数

### #23 · 2026-10-27 · 空间交叉验证实验设计（只设计，不训练）
- 意图：在不执行训练的前提下，核验 18 个 `cv_fold` 的空间有效性，给出能同时保证时间隔离与空间隔离的 CV 方案，并明确指标、不确定性与异常处理
- 改动（独立输出）：
  - **新建** `scripts/profile_cv_folds_20261027.py`：折有效性 9 项核查 + 逐折 × 阶段画像 + 冻结模型分折指标（只读）
  - **新建** `outputs/spatial_cv_design_20261027/`：`spatial_cv_design.md`、`fold_profile.csv`、`frozen_test_fold_metrics.csv`、`cv_checks.json`
- 验证：
  - V1 通过：18 折 ↔ 18 个 `dc` 双射；V2 通过：排除 golden 后 5,473 网格跨折 0（上一轮「75 格跨折」全部来自 936 条 golden 被置 -1）
  - V3 通过：`cv_fold=-1` 41,964 行 = 44 个无区网格 41,028 行 + golden 936 行，整组排除
  - V4/V6 警示：折规模 35–991 网格、15,948–546,948 行；正例 <50 的折为 1 与 6（训练正例 235/47）
  - V7 警示：56 个正例日期每折全覆盖 → 空间折不能隔离事件日，必须叠加时间切分
  - V9 警示：冻结 dual_xgb 分折测试 AUC 0.589–0.829（mean 0.733、SD 0.066），static_only 0.466–0.870（1 折低于 0.5），整体 0.751 掩盖区域差异
  - 设计要点：LODO-T（训练 2018–2021 排除折 k；阈值用留一折池化 2022；测试 2023–2026 ∩ 折 k）；3 模型（dual_xgb / static_only_xgb / dual_lr）；分折分布 + 折级 bootstrap CI + 配对检验；无正例/异常/失败处理规则；估算 15–30 min、6–10 GB
- 遗留：**未执行训练**，等待用户批准；2022 验证段仅 2 个事件日，阈值证据薄弱已写入限制

### #24 · 2026-10-27 · T6 发布前检查与最小修复（GitHub 发布准备）
- 意图：在不公开仓库的前提下，对 18 区气象播报做发布前检查，输出阻塞项清单，并补齐零风险的展示/文档缺口
- 改动：
  - **更新** `hko_api/district/index.html`：注明雨量分级为本产品展示规则、非天文台官方分级
  - **新建** `hko_api/district/README.md`：运行步骤、测试命令、字段口径、异常处理、限制与来源声明
  - **更新** `.gitignore`：排除 `hko_api/data/raw/`（276 MB / 2,837 文件）
  - **新建** `hko_api/district/reports/t6_release_checklist.md`：10 项检查 + 4 个阻塞/待确认项 + 发布前命令清单
- 验证：`node hko_api/district/tests/bulletin.test.mjs` → 8 passed / 0 failed；`pytest` 167 项通过；敏感信息扫描无 key/token/私钥；`hko_api/data/latest` 1.7 MB、`district` 48 KB，可入库
- 遗留/阻塞：B2 公开快照 30 分钟后必显示 STALE（需选择演示口径）；B3 HKO 数据再分发条款待确认；B4 仓库零提交、无 LICENSE；B6 移动端未真机实测（截图工具不可用）
