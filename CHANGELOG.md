# CHANGELOG

本项目从 2026-10-26 起按用户任务书要求（§三.6）维护本文件；此前历史见 `ITERATIONS.md` 与 `reports/`。
每次生成新数据版本、改变研究口径或修复影响结果正确性的问题时，在此追加条目。

---

## [delivery_repaired_b1] · 2026-10-26 · B 侧独立标签修复（R2 方案）

**背景**：D1 审计（`reports/warning_date_authenticity_audit.md`）确认 A 侧警告解析混入 320 天红色火灾危险警告，95.3% 正例落在假事件日。用户批准 R2 方案：B 侧基于 HKO 官方 `rstorm.dat` 独立重建标签与抽样框架。

**新增**
- `data/delivery_repaired_b1/flood_risk_dataset_2h_500m.parquet`（2,905,896 × 109，SHA256 `1749cc00…5cb4e8`）
- `data/delivery_repaired_b1/sample_frame.parquet`（SHA256 `423d1e85…1f47d79b`）
- `data/delivery_repaired_b1/manifest_repair_b1.json`（来源/哈希/种子/参数/差异统计全记录）
- `scripts/b_side_repair.py`（feasibility / repair / validate 三子命令，seed=42）

**变更（相对 delivery_v10_20261025）**
- 警告源：未核验解析 → HKO 官方 `rstorm.dat`（raw SHA256 `c726d4ba…b09f45b`），事件日 376 → 56（官方红 42 + 黑 14）
- 重建 10 个标签列：warn_level、warn_level_day、is_event_day、flooded、flooded_2h、flooded_2h_peak、label_source、confidence、attrib_conf、sample_role；59 个模型特征与其余 89 列逐行未动
- 320 个假事件日改判非事件日并重抽 35 格/日（受限重抽：仅能从当日已有 2,362 格中选，A 类占比 55.5% 偏差已记录）
- 顺带纠正旧数据 bug：200 个真实 amber 日 `warn_level_day` 0→1（84,000 行，EXCLUDED 列）
- 黄金事件 936 行标签逐值保留；诊断集只读未动

**验证**：11/11 通过（`reports/b_side_repair_validation.md`，`outputs/repair_b1/validation.json`）——主键唯一、事件日=官方、红黑槽映射 0 错、正例来源违例 0、计数可追溯、时间覆盖完整、黄金保留、特征逐值不变、划分互斥、诊断集隔离、局限记录。

**报告**：`reports/b_side_repair_feasibility.md`、`b_side_repair_validation.md`、`b_side_old_new_comparison.md`。

**注意**：正例率 35.67% → 6.77% 系官方警告推演的自然结果，非调节目标；新旧数据上模型指标不可直接比较（`b_side_old_new_comparison.md` §8）。正式全量训练尚待用户批准。

---

## [formal_training_repaired_b1] · 2026-10-26 · 修复版数据正式全量训练（用户 9 项条件批准）

**背景**：`delivery_repaired_b1` 通过 11/11 验证与 30 万轻量实验后，用户批准正式训练。数据版本与哈希冻结（SHA256 `1749cc00…5cb4e8`），全程未覆盖任何旧数据/模型/报告。

**新增**
- `scripts/run_formal_training_repaired_b1.py`（--preflight 六道中止闸门 + 正式训练：5 模型、val 阈值扫描、SHAP、GIS、图件、报告生成）
- `outputs/formal_training_repaired_b1_20261026-020814/`：metrics.json、5 个模型 bundle、`gis/grid_risk_500m.csv`（5,517 格，QGIS 联接 `grid_id`）、predictions_val/test.parquet、5 份阈值扫描表、shap_importance.csv、6 张图件
- `reports/formal_training_results_repaired_b1.md`（模型对比、阈值依据、消融、SHAP、GIS、三项偏差披露、论文可用段落）

**模型与口径**
- train=2018–2021（1,199,472 行）/ val=2022（209,148）/ test=2023–2026.09（1,496,340，仅最终评估）；golden 仅敏感性参考
- 5 模型无调参：dual_xgb（主，59 特征）、dual_lr（基线）、power_only_xgb（对照）、static_only_xgb / rain_time_only_xgb（消融）
- 阈值：每模型在 val 上 F1 最大点选定（dual_xgb τ=0.05，dual_lr τ=0.89）；旧 0.3 阈值弃用（在修复版上 F1 仅 0.014）
- test 上：dual_xgb ROC 0.751 / PR 0.184 / F1 0.241；dual_lr ROC 0.750 / PR 0.291 / F1 0.293；ERA5 增益 +0.027 ROC

**关键发现与局限（详见报告 §5.2）**：val 仅 2 个红黑事件日导致 val ROC 0.961 → test 0.751 泛化差；静态地形记忆主导判别力（静态单独 0.720，动态降雨边际 +0.031）；事件日内空间分辨 ROC 0.638；18 折空间 CV 未执行（用户允许）。全部结果为弱监督标签下的探索性结果，模型输出是风险评分，不得称为已验证的真实水浸发生概率。

---

## [supplement_tide_ablation_b1] · 2026-10-26 · 最小补充实验：去潮位消融 + 一致性 + 动态能力 + 阈值敏感性 + 雨潮复合（用户 5 项批准）

**背景**：正式训练后用户批准的最小补充实验；硬约束为不做 18 折空间 CV、不做大规模调参、不做复杂概率校准；全部产物独立目录，未覆盖任何旧版本，未启动其他训练任务。

**新增**
- `scripts/run_supplement_tide_ablation_b1.py`（八步：冻结哈希 → no_tide 训练 → 对比 → 一致性 → 动态能力 → 阈值敏感性 → 雨潮复合 → 落盘）
- `outputs/supplement_tide_ablation_b1_20261026-024612/`：metrics.json、results_report.md、no_tide_xgb bundle、predictions_test_no_tide.parquet、PR-AUC 诊断（xgb vs lr）、校准十分位表、沿海分组表、雨潮复合表（事件日/全 test）、阈值敏感性表×2
- `reports/supplement_tide_ablation_b1.md`（实验报告）、`reports/paper_updates_ch3_ch4_b1.md`（论文三/四章行号级修改清单）

**关键结果**
- 去潮位消融：no_tide_xgb（移除潮汐组 4 列，泄漏核查剩余 55 列不含 tide/surge/tidal）test ROC 0.7579（Δ+0.0070）、PR 0.1929（Δ+0.0093）→ 判别力不由潮位贡献
- 一致性：正式 bundle 重预测概率逐值 diff=0.0e+00（dual_xgb / dual_lr 均是），y 逐行一致；LR PR-AUC 优于 XGB 系顶端排序更尖（precision@1000 0.776 vs 0.407）+ balanced 权重概率尺度失真，未重调参
- 动态能力：dual_xgb 全样本 ROC 0.751 / 事件日内 0.638 / 日期级 0.897 / grid-day 级 0.950；沿海组 0.576 弱于内陆 0.802 → 强在日期级预警与静态空间分辨，弱在事件日内小时级分辨
- 阈值敏感性（仅 test 敏感性分析，未重选阈）：旧 0.30 阈值 Recall 仅 0.072；维持 val 选定 τ=0.05
- **雨潮复合（关键发现）**：标签层「雨高×潮高」正例率 22.90%（vs 其余 12.78%，沿海 41.88%）但模型预测概率反向更低（0.0281 vs 0.0499）→ 弱标签存在雨潮复合正例聚集、模型未学到该交互；论文仅可表述为假设与探索方向

**验证**：pytest 167 项通过；数据 SHA256 `1749cc00…5cb4e8` 未变。

---

## [final_verification_spatial_b1] · 2026-10-26 · 最终核验 + 空间应用分析（只读，用户 5 项批准）

**背景**：补充实验后的最后一轮只读核验；不训练、不改原始数据与旧报告。全部复算与既有产物一致（聚合 AUC diff=0.00e+00，雨潮复合 mean_p diff≤4.9e-07）。

**新增**
- `scripts/run_final_verification_spatial_b1.py`（SHA 冻结 + 四项核验，全程只读）
- `outputs/final_verification_spatial_b1_20261026-030632/`（metrics.json、results_report.md、rain_tide_composite_recheck.csv 含逐组合正例数与 mean_is_A、top20_grid_rank_comparison.csv）
- `reports/final_verification_spatial_b1.md`（三层结论清单：直接支持 / 谨慎解释 / 无法证明）

**核验结论**
- 日期级 0.897 / 格日级 0.950 为回顾性日级聚合（p_max 含事件后槽位，日内信息回流；标签为事后确定的日级警告属性），**不能支持"提前 2 小时动态预警"**；符合 2h 口径的仅事件日内行级 ROC 0.638；格日级 0.950 系容易任务（基线 34.7%）
- 雨潮组合：各箱 mean_is_A 恒定（0.552–0.557）→ "雨高×潮高 22.90%"非 is_A 组成机械造成；但雨维非单调（雨高×潮中最低 6.74%）且模型评分最高组合恰为正例率最低之一 → 聚集受警告归因窗口槽级时序结构影响，仍只可作假设表述
- 空间排名（test 期逐格聚合，Spearman 0.9888，Top20 重合 9/20，头部概率差 <0.008）：去潮位不改变空间格局
- **GIS `mean_risk_all` 证实为全期 train+val+test 非 golden 逐格平均**（行数复核 5,517/5,517），含训练期数据；论文引用必须标注口径，test-only 排名以 top20_grid_rank_comparison.csv 为准
- **未生成社区级优先级排名**：交付数据无社区/街道边界映射；人口/脆弱性字段为格级常量且按设计未进模型

**验证**：pytest 167 项通过；数据 SHA256 `1749cc00…5cb4e8` 未变。

---

## [paper_package_b1] · 2026-10-26 · 论文成果整理包（只读，不训练）

**背景**：最终核验后的论文整理阶段；停止全部训练与实验，仅整理既有成果。数据 SHA256 `1749cc00…5cb4e8` 未变，论文原文未修改。

**新增**
- `scripts/run_paper_package_b1.py`（只读整理：对比表数值直取冻结 metrics.json 不重算）
- `outputs/paper_package_b1_20261026-122643/`
  - `tables/final_model_comparison.csv`（6 模型全指标 + delta_vs_dual）
  - `tables/grid_risk_test_period_500m.csv`（测试期逐格 mean/max + 排名 + Top20 标志，QGIS 可联接；mean_risk_all 仅作参考列）
  - `figures/`：fig_paper_roc_pr_test.png（6 模型 ROC+PR）、fig_paper_rain_tide_heatmap.png（3 口径×标签层/模型层）、fig_paper_grid_risk_test_vs_full.png（测试期 vs 全期并排、口径在标题区分）、fig_paper_rank_stability.png（Spearman 0.989、Top20 9/20）、fig_paper_shap_top20.png（tide_pred_max_m 列第 11、非主导）
  - `formal_originals/`：正式训练 6 张原图备查
- `reports/paper_material_ch3_ch4_b1.md`（三/四章素材包：方法/数据来源/弱监督标签/样本划分/口径分层指标/局限 + 图注建议 + 三层引用速查）
- `reports/paper_paragraph_audit_b1.md`（全文逐段审计：27 条不一致 + 16 条需加限定；最严重 5 条 = L88 标签口径、L402 结论导语三重不实、L403 雨潮写成已验证结论、L404 社区级产出虚构、L408 三项披露缺失）
- `reports/paper_updates_ch3_ch4_b1.md` §1 加更正标注：旧稿分年 ROC 誊录错误，正确值 0.7442/0.8311/0.7752/0.7220（metrics.json by_year）

**口径要点**
- GIS 全期 vs 测试期差异实证：全期 top1 H078033（0.1149）在测试期列 21；测试期 top1 H092081（0.0990）全期列 29——论文实证展示优先用测试期，引用全期必须标注"含训练期数据"
- 日期级 0.897 / 格日级 0.950 仍仅作回顾性聚合表述；2h 口径唯一对应事件日内 ROC 0.638
- 无社区边界映射，未生成任何社区级排名

**验证**：pytest 167 项通过；对比表/Spearman/Top20/审计关键数字与冻结产物逐一核对一致。

