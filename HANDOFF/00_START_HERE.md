# 00 · START HERE — 香港智慧防洪项目交接入口

> 生成日期：2026-10-27　｜　生成方式：对当前仓库全量只读核查后编写
> 本目录（`HANDOFF/`）是团队技术交接包，供新协作者快速了解项目状态、约束和下一步工作。

---

## 1. 项目是什么

**香港社区水浸（暴雨内涝）风险预测系统**，筑工杯「智慧防洪」项目 B 侧（建模侧）产物。

- **综合架构**：气象信息 + 水浸风险建模（XGBoost）+ GIS 空间治理分析
- **空间单元**：香港全域 5,517 个 500 m 规则网格（EPSG:2326 抽陆）
- **时间单元**：2 小时槽（每日 12 槽），行 = `(grid_id, date, slot)`
- **预测任务**：槽开始时刻 t 发起，预测 `[t, t+2h)` 窗口内是否水浸（弱监督标签）
- **双轨定位**（`PROJECT_PRINCIPLES.md`，v1.0 长期有效）：
  - **研究轨道**：支撑论文交付，产物冻结、可追溯、可复现
  - **产品轨道**：HKO 实时气象接入、18 区播报、GIS Dashboard 等持续开发

## 2. 当前状态一句话

**研究主线已完成并冻结**（修复版数据 + 6 模型正式训练 + 论文图表包 + 四表填充），**产品 MVP 已交付**（HKO 五源接入 + Dashboard + 18 区播报页）。当前私有 GitHub 仓库用于团队协作；正式数据、模型与论文产物不随 Git 仓库分发，获取方式见根目录 `CONTRIBUTING.md`。

## 3. 最重要的五条事实（新协作者必读）

1. **所有模型指标是弱监督标签下的探索性结果**，标签 = A 类空间规则 × HKO 官方红/黑暴雨警告窗口（`rstorm.dat` 核验，56 个官方事件日），**不是实测水浸记录**；模型输出只能称「弱监督相对风险评分」，**严禁**称为「真实水浸概率」。
2. **冻结产物不可覆盖**：`data/delivery_repaired_b1/`（SHA256 `1749cc00…5cb4e8`）、`outputs/formal_training_repaired_b1_20261026-020814/`、`outputs/supplement_tide_ablation_b1_20261026-024612/`、`outputs/paper_package_b1_20261026-122643/`。新实验必须写入新的带日期目录。
3. **主模型 dual_xgb 测试期 ROC-AUC 0.751 / PR-AUC 0.184**；判别力主要来自静态地形记忆（纯静态 0.720），动态降雨边际 +0.031 —— 模型是「已知地形下的风险空间分辨」，**不是**降雨驱动的 2 小时动态预警（事件日内 ROC 仅 0.638）。
4. **实时与历史严格隔离**：`hko_api/data/` 为实时归档；16 个再分析降雨特征（POWER/ERA5 prev 族）**无法**由现有实时 API 原口径替代；实时 XGBoost 推理**未实施、未批准**。
5. **任何训练、调参、标签修改、阈值变更、空间 CV 执行都必须经项目负责人明确批准**，不得自动进行。

## 4. 如何启动（最小路径）

```bash
# 环境（macOS，Python 3.12；详见 07_ENVIRONMENT_SETUP.md）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 验证基线（应为 167 passed）
python -m pytest

# 产品 MVP（两个终端）
.venv/bin/python -m hko_api.hko_fetch --loop          # 实时抓取
.venv/bin/python -m http.server 8321 --bind 127.0.0.1 # 静态服务
# 浏览器打开 http://127.0.0.1:8321/hko_api/dashboard.html
```

## 5. 推荐阅读顺序

| 顺序 | 文件 | 读它是为了什么 |
|---|---|---|
| 1 | `HANDOFF/05_DECISIONS_AND_CONSTRAINTS.md` | 先知道什么**不能做** |
| 2 | `PROJECT_PRINCIPLES.md` | 双轨规范全文（人读版） |
| 3 | `HANDOFF/03_MODEL_RESEARCH.md` | 研究成果与冻结产物索引 |
| 4 | 团队共享材料中的 `reports/paper_delivery_checklist.md` | 论文交付项 已完成/待验证/缺失（不在 Git 仓库中） |
| 5 | `HANDOFF/01_ARCHITECTURE.md` | 代码与数据流全貌 |
| 6 | `ITERATIONS.md` | 24 条迭代记录（全部历史决策的来龙去脉） |
| 7 | `README.md` | 建模框架使用说明 |

## 6. HANDOFF 目录文件清单

| 文件 | 内容 |
|---|---|
| `00_START_HERE.md` | 本文件 |
| `01_ARCHITECTURE.md` | 技术架构、模块、数据流 |
| `02_DATA_INVENTORY.md` | 全部数据集清单（路径/来源/规模/哈希/授权） |
| `03_MODEL_RESEARCH.md` | 模型、标签、特征、指标、SHAP、空间 CV、冻结位置 |
| `04_PRODUCT_STATUS.md` | HKO API / Dashboard / 18 区播报 / STPU 分析产品状态 |
| `05_DECISIONS_AND_CONSTRAINTS.md` | 已确认决策、研究红线、审批机制 |
| `06_PENDING_TASKS.md` | 未完成任务（P0/P1/P2，研究/产品分列） |
| `07_ENVIRONMENT_SETUP.md` | 环境、依赖、启动、测试、故障 |
| `FILE_MANIFEST.csv` | 关键文件与冻结研究产物清单（部分产物不在 Git 中） |

## 7. 本地服务说明

仓库不记录任何协作者机器上的 PID 或运行状态。启动、停止和端口排查命令见 `04_PRODUCT_STATUS.md` §6；每位协作者应只管理自己机器上的进程与 `hko_api/data/` 归档。
