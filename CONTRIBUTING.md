# 团队协作指南

本仓库用于项目组成员阅读、修改和审查代码。开始工作前，请先阅读 `HANDOFF/00_START_HERE.md`；涉及模型、标签或正式结果时，再阅读 `PROJECT_PRINCIPLES.md` 与 `HANDOFF/05_DECISIONS_AND_CONSTRAINTS.md`。

## 1. 首次开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest
node hko_api/district/tests/bulletin.test.mjs
```

建议从新分支工作，提交内容保持单一主题，并通过 Pull Request 让另一位同学复核。普通代码、文档和测试改进可以直接提出 PR；重新训练、修改标签、替换冻结结果或改变正式阈值前，先在团队内确认方案。

## 2. 文档入口

| 想了解什么 | 先读哪里 |
|---|---|
| 项目概览、当前能力与限制 | `README.md` |
| 交接入口和推荐阅读顺序 | `HANDOFF/00_START_HERE.md` |
| 架构与数据流 | `HANDOFF/01_ARCHITECTURE.md` |
| 数据版本、来源与缺口 | `HANDOFF/02_DATA_INVENTORY.md` |
| 模型、指标与解释边界 | `HANDOFF/03_MODEL_RESEARCH.md` |
| Dashboard、HKO API 和产品状态 | `HANDOFF/04_PRODUCT_STATUS.md` |
| 研究红线与已确认决策 | `HANDOFF/05_DECISIONS_AND_CONSTRAINTS.md` |
| 下一步任务 | `HANDOFF/06_PENDING_TASKS.md` |
| 环境、命令和故障排查 | `HANDOFF/07_ENVIRONMENT_SETUP.md` |
| 历史变化 | `CHANGELOG.md`、`ITERATIONS.md` |

## 3. 仓库外材料

以下内容因体积、授权或隐私原因不纳入 Git：

- 正式训练数据和第三方派生数据；
- 模型 bundle、测试集预测、SHAP、GIS 评分与论文图表包；
- 内部研究报告和论文交付文件；
- HKO 原始归档与运行时最新快照。

需要这些材料时，由项目负责人通过团队共享存储提供。收到后先核对 `HANDOFF/FILE_MANIFEST.csv` 中的版本和 SHA256，不要自行从不明来源补文件，也不要把共享材料重新提交到 Git。

没有正式数据时，仍可使用合成配置、单元测试和前端规则测试开发通用功能。历史评分 Dashboard 还需要 `hko_api/data/latest/grid_scores.json|js`；若缺失，气象抓取功能仍可独立运行。

## 4. 提交前检查

```bash
git status --short
python -m pytest
node hko_api/district/tests/bulletin.test.mjs
```

同时确认提交中不含 `.env`、密钥、个人电脑绝对路径、`data/`、`outputs/`、`reports/`、运行时快照或压缩归档。模型指标必须注明数据版本、切分口径和弱监督标签限制。
