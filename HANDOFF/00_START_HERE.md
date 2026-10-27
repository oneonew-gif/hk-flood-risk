# 00 · START HERE — 香港智慧防洪项目交接入口

> 生成日期：2026-10-27　｜　生成方式：对当前仓库全量只读核查后编写
> 本目录（`HANDOFF/`）为跨平台迁移交接包，供新 Agent 接手。除本目录外未修改仓库任何文件。

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

**研究主线已完成并冻结**（修复版数据 + 6 模型正式训练 + 论文图表包 + 四表填充），**产品 MVP 已交付**（HKO 五源接入 + Dashboard + 18 区播报页），仓库 **git 零提交、无远程、无 LICENSE**，论文尚有若干「待验证/缺失」项（见 `reports/paper_delivery_checklist.md`）。

## 3. 最重要的五条事实（新 Agent 必读）

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
| 4 | `reports/paper_delivery_checklist.md` | 论文交付项 已完成/待验证/缺失 |
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
| `08_GITHUB_RELEASE_AUDIT.md` | GitHub 发布适配性检查（仅检查，不执行） |
| `09_AGENT_INSTRUCTIONS.md` | 新 Agent 的长期工作规则 |
| `FILE_MANIFEST.csv` | 关键文件清单（路径/用途/大小/SHA256/冻结状态） |

## 7. 交接时的运行中进程（2026-10-27 14:40 实测）

| 进程 | 启动时间 | 说明 |
|---|---|---|
| `.venv/bin/python -m hko_api.hko_fetch --loop`（PID 69070） | 2026-10-26 下午 | HKO 五源轮询抓取，持续写入 `hko_api/data/` |
| `.venv/bin/python -m http.server 8321 --bind 127.0.0.1`（PID 66215） | 2026-10-26 中午 | Dashboard 本地服务 |
| `python -m http.server 8322 --bind 127.0.0.1 --directory /tmp/t6_fixtures`（PID 49015） | 2026-10-27 14:10 | T6 测试夹具临时服务（可忽略） |

> 迁移后如需在新环境重启，先确认旧机器上这两个长期进程已停止，避免端口与归档目录冲突。停止/重启方法见 `04_PRODUCT_STATUS.md` §6。
