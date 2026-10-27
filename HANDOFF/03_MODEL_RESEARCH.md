# 03 · 模型与研究成果

> 全部数值取自冻结产物与审计报告（2026-10-27 复核一致）。
> **总口径声明**（metrics.json disclaimer 原文要点）：标签 = is_A × 官方红/黑警告归因窗口（HKO rstorm.dat），全部指标**不得解释为真实水浸提前 2 小时预测准确率**，模型输出**不得称为已验证的真实水浸发生概率**；样本含 320 个改判日受限重抽偏差；ERA5/POWER 为历史再分析口径，与实时部署输入存在时效差异。

---

## 1. 任务定义

- **行**：`(grid_id, date, slot)`，5,517 格 × 2h 槽 × 2018-01-01~2026-09-30
- **锚定**：行时间戳 = 槽开始 = 预测发起时刻 t；主目标 `y_future_2h` 不位移
- **标签**：`flooded_2h` = 1 当且仅当 `[t, t+2h)` 落入弱监督正例规则（A 类空间规则 × 官方红/黑暴雨警告窗口）；黄金事件 936 行标签逐值保留
- **正例率**：全表 6.77%（196,693 / 2,905,896）；test 8.49%

## 2. 数据划分（固定，不得改）

| 集合 | 时间 | 行数 | 用途 |
|---|---|---|---|
| train | 2018–2021 | 1,199,472 | 训练 |
| val | 2022 | 209,148 | 阈值选择（仅 2 个红黑事件日，789 正例，证据薄弱已披露） |
| test | 2023-01 ~ 2026-09 | 1,496,340 | **仅最终评估**；未参与调参与选阈 |
| golden_eval | 2 个事件 | 936 | 仅敏感性参考（加密样本） |

## 3. 特征（59 个，`src/floodrisk/realdata.py` FEATURE_GROUPS）

- **静态 40**：地形（elevation/slope/tpi/roughness/hydro1km_*…）、土地覆盖 esa_lc_*、不透水面、排水距离、最近潮位站距离等
- **降雨·前期累计 16**：POWER 8（prev3h/6h/12h/24h/48h、max1h_prev6h/24h、api72）+ ERA5 8（同结构）；**方案 A**：两源并列、保留来源标识、不合并
- **潮位预报 4**：tide_pred_max_m / tide_pred_mean_m / tide_n_hours / dist_to_tide_m（天文潮为确定性预报，t 时刻完全可知，无泄漏）
- **时间 3**：month / hour_sin / hour_cos
- **EXCLUDED 57 列**：标识、标签族、is_A 规则内部列、抽样/黄金标记、split/cv_fold/dc、dist_to_blackspot/lowlying 等，逐条带理由
- 59 ∩ EXCLUDED = ∅ 已校验

## 4. 正式模型（全部无调参，seed=42，`outputs/formal_training_repaired_b1_20261026-020814/bundles/`）

| 模型 | 特征 | test ROC-AUC | test PR-AUC | Brier | F1 @ τ(val 选定) | bundle SHA256（前 12 位） |
|---|---|---|---|---|---|---|
| **dual_xgb（主）** | 59 | **0.7509** | 0.1837 | 0.0801 | 0.241 @ 0.05（P 0.192 / R 0.324） | `0a8440106a06…` |
| dual_lr（基线） | 59 | 0.750 | **0.291** | — | 0.293 @ 0.89 | `836fd023cab2…` |
| power_only_xgb（对照） | 51 | 0.724 | — | — | @ 0.05 | `3c96009a98c9…` |
| static_only_xgb（消融） | 40 | 0.720 | — | — | @ 0.07 | `54cd905b27aa…` |
| rain_time_only_xgb（消融） | 19 | 0.647 | — | — | @ 0.03 | `53c9b3e3e272…` |
| no_tide_xgb（补充消融） | 55 | 0.7579 | 0.1929 | −0.0009 Δ | @ 0.04 | `9aab0649728d…`（在 supplement 目录） |

- **LR 与 XGB 比较如实保留**：LR PR-AUC 优于 XGB（0.291 vs 0.184），机理已定位（LR 顶端排序更尖 precision@1000 0.776 vs 0.407 + XGB balanced 权重概率尺度失真），**未重调参**；完整 6 模型 18 字段对比见 `outputs/paper_package_b1_20261026-122643/tables/final_model_comparison.csv`
- **ERA5 增益**：dual vs power_only ROC +0.027
- **golden 敏感性**：ROC 0.904（仅供参考）
- **阈值纪律**：旧 0.30 阈值在修复版上 F1 仅 0.014，已弃用；现有 τ 均为 val 上 F1 最大点；test 阈值敏感性仅作补充说明（`tables/threshold_sensitivity_test.csv`），不得用 test 重选阈

## 5. 能力分层（论文口径，必须分层引用）

| 口径 | dual_xgb ROC | 能否称「提前 2h 预警」 |
|---|---|---|
| 全样本（test） | 0.751 | 否（含大量非事件日易分辨样本） |
| **事件日内行级（唯一符合 2h 口径）** | **0.638** | 这是 2h 动态分辨能力的上限表述 |
| 日期级聚合 | 0.897 | **否** —— p_max 含事件后槽位（日内信息回流），回顾性口径 |
| 格日级聚合 | 0.950 | **否** —— 同上，且为容易任务（基线 34.7%） |
| val 2022 | 0.961 | 否 —— val 仅 2 个事件日，val→test 泛化差显著 |
| 沿海组 / 内陆组 | 0.576 / 0.802 | 沿海明显弱 |

## 6. 关键研究结论（已冻结，论文可用）

1. **静态地形记忆主导判别力**：static_only 0.720 vs dual 0.751，动态降雨边际 +0.031；当前模型是「已知地形下的风险空间分辨」而非降雨驱动动态预测
2. **去潮位不降级**：no_tide ROC 0.7579（Δ+0.0070）→ 判别力不由潮位贡献
3. **雨潮复合方向相反**：标签层「雨高×潮高」正例率 22.90%（其余 12.78%，沿海 41.88%），但模型预测概率反而更低（0.0281 vs 0.0499）→ 弱标签存在复合正例聚集、模型未学到交互；**论文仅可表述为假设与探索方向**
4. **空间格局稳健但头部名次敏感**：dual vs no_tide 逐格排名 Spearman 0.9888，Top20 重合 9/20
5. **GIS 口径**：`gis/grid_risk_500m.csv` 的 `mean_risk_all` 为**全期 train+val+test 非 golden 逐格平均**（含训练期）；测试期口径以 `paper_package.../tables/grid_risk_test_period_500m.csv` 为准。全期 top1 H078033(0.1149) 测试期列 21；测试期 top1 H092081(0.0990) 全期列 29 —— 两口径不可混用

## 7. SHAP（dual_xgb，xgboost pred_contribs 精确 TreeSHAP）

- 文件：`outputs/formal_training_repaired_b1_20261026-020814/shap_importance.csv`（SHA256 `32938706…`）
- 全局 Top：elevation 居首；`tide_pred_max_m` 列第 11（0.2493），非主导
- 图：`figures/fig_shap_bar.png`、`fig_shap_dependence.png`；论文图 `paper_package.../figures/fig_paper_shap_top20.png`
- **边界**：SHAP 仅解释模型输出的特征贡献，**不得写成已确认的实际水浸成因**；个案/区域级 SHAP 未计算（缺失）

## 8. 空间交叉验证（待验证 · 未执行）

- 设计文档：`outputs/spatial_cv_design_20261027/spatial_cv_design.md`（LODO-T：训练 2018–2021 排除折 k；阈值留一折池化 2022；测试 2023–2026 ∩ 折 k）
- 折有效性已核：18 折 ↔ 18 个 dc 双射；5,473 网格跨折 0（golden 936 强制 -1）；折规模 35–991 格；折 1/6 正例 <50
- **警示**：56 个正例日期每折全覆盖 → 空间折不能隔离事件日，必须叠加时间切分
- 冻结模型分折 test AUC 0.589–0.829（mean 0.733，SD 0.066）→ 整体 0.751 掩盖区域差异
- **18 折完整 CV 未执行，论文不得写成已完成；执行需用户批准**（估算 15–30 min / 6–10 GB）

## 9. 冻结文件位置汇总（研究轨道）

| 类别 | 路径 |
|---|---|
| 训练数据 | `data/delivery_repaired_b1/`（3 文件，哈希见 02） |
| 正式模型 bundles ×5 | `outputs/formal_training_repaired_b1_20261026-020814/bundles/` |
| 补充模型 no_tide_xgb | `outputs/supplement_tide_ablation_b1_20261026-024612/bundles/` |
| 指标/报告 | 同两目录 `metrics.json`、`results_report.md` |
| 测试集预测 | `formal_.../predictions_test.parquet`、`supplement_.../predictions_test_no_tide.parquet` |
| SHAP | `formal_.../shap_importance.csv` |
| GIS 评分 | `formal_.../gis/grid_risk_500m.csv`（全期）、`paper_package_.../tables/grid_risk_test_period_500m.csv`（测试期） |
| 论文图表包 | `outputs/paper_package_b1_20261026-122643/`（18 文件 + MANIFEST.sha256） |
| 论文逐段审计/素材 | `reports/paper_paragraph_audit_b1.md`、`paper_material_ch3_ch4_b1.md`、`paper_updates_ch3_ch4_b1.md` |
| 四表填充 | `outputs/paper_tables_fill_20261026/`（含 docx 审核副本 v2） |

## 10. 模型局限（论文限制章节口径）

1. 弱监督标签 ≠ 实测水浸；事件日标签与降雨曾严重脱钩（旧版 81% 事件日全港雨量 <1mm，已通过 R2 修复为官方警告源，但弱监督本质不变）
2. 静态主导 → 动态预警能力弱（事件日内 0.638）
3. val 仅 2 个事件日，阈值证据薄弱
4. 320 改判日受限重抽（仅能从当日已有 2,362 格中选，A 类占比 55.5%）抽样偏差
5. 16 个再分析降雨特征部署期不可原口径实时获取（distribution shift）
6. 无社区/区议会边界 → 未生成任何社区排名；人口/脆弱性未进模型（按设计）
7. 随机森林基线**未训练**（早期 300k 实验中 RF 曾优于 XGB，但属旧标签体系，不可引用为正式结果）；论文表中 RF 行必须留空或标「未训练」
