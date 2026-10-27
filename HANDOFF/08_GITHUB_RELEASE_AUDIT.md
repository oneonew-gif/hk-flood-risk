# 08 · GitHub 发布适配性检查（仅检查，未执行任何上传）

> 检查日期：2026-10-27。依据：本次全库只读扫描 + `hko_api/district/reports/t6_release_checklist.md`（T6 发布前检查）。
> **结论：当前不适合直接公开。存在 4 个阻塞项（B1–B4）。**

---

## 1. 仓库基线

| 项 | 实测状态 |
|---|---|
| git 初始化 | ✅ 已初始化（分支 main） |
| 提交 | ❌ **零提交**，全部文件 untracked |
| 远程 | ❌ 无 remote |
| LICENSE | ❌ 无 |
| README | ✅ 存在（但面向建模框架，缺少项目总览与产品入口说明） |
| CI | ❌ 无 |

## 2. 敏感信息扫描

- 全库扫描（`.env` / `*.key` / `*.pem` / `secrets*` / API key / token 模式）：**未发现**
- HKO 为开放数据，无需密钥；代码中无硬编码凭证
- ⚠️ 桌面存在 `github-recovery-codes.txt`（仓库外，与本仓库无关，但提醒用户勿误传）
- ⚠️ 报告与文档中含真实人名（数据侧维护者姓名，见 `data/delivery_v10_20261025/README.md`）与学校课程编号 —— 公开发布前需用户确认是否脱敏

## 3. 大文件与数据授权

| 类别 | 现状 | .gitignore 覆盖 |
|---|---|---|
| `data/` 全部真实/合成数据 | 1.2 GB+（v10 系列 401+378+259 MB） | ✅ `data/*.parquet`、`data/*.csv` 等已排除 |
| `outputs/` 全部实验产物 | ~330 MB（含 51 MB 正式训练、模型 bundle） | ✅ `outputs/` 整体排除（模型 `*.joblib` 亦单独排除） |
| `hko_api/data/raw/` | 280 MB / ~2,800 文件 | ✅ 已排除（T6 修复） |
| `hko_api/data/latest/` | 1.7 MB | 未排除，可入库（但见 B2） |
| `model_demo.html` | 245 KB | 可入库 |

- **可复现性矛盾（需用户决策）**：`.gitignore` 排除了全部数据与 outputs —— 公开仓库将只有代码与文档，**任何人无法复现论文结果**（数据无公开授权，属预期；但 README 需写明「数据不公开」）。
- **HKO 数据授权（B3）**：开放数据可引用，但**再分发条款未核实**；`hko_api/data/latest/snapshot.json` 是否可入公开库待确认（备选：运行时抓取、不入库）。
- **A 方数据授权**：交付数据含 2021 普查派生、WorldPop、ESA 土地覆盖、NASA POWER、ERA5 等第三方来源的再加工产物 —— 公开任何派生数据前需逐一确认上游许可。

## 4. 阻塞项汇总（发布前必须处理）

| # | 阻塞项 | 说明 |
|---|---|---|
| B1 | ~~raw 归档体积~~ | ✅ 已通过 .gitignore 处理（T6） |
| B2 | **演示快照 30 分钟后必显示 STALE** | 公开库携带冻结 snapshot，首访者会看到「过期」横条；需在 README 注明演示口径 / 不入库 latest / 或 Actions 定时刷新（三选一，**未决定**） |
| B3 | **HKO 数据再分发条款未核实** | 发布前确认 |
| B4 | **零提交、无 LICENSE** | 至少需首次提交 + main 基线 + LICENSE |
| B5 | 无 CI | 建议最小 CI：`node hko_api/district/tests/bulletin.test.mjs` + `pytest` |
| B6 | 移动端未真机实测 | 375px 视口人工复核 |
| B7 | 人名/课程信息脱敏确认 | 见 §2 |

## 5. 论文相关特殊考量（本项目独有）

- 仓库含比赛/论文未完成品：公开后**弱监督标签的局限披露必须随库发布**（README 首段 + 论文 checklist 链接），否则模型评分易被误读为「真实水浸概率」
- `outputs/` 不入库意味着论文图表包不在 GitHub；如需支撑评审，建议单独 Release 附件（需用户决定）
- `HANDOFF/` 本目录含项目内部决策与遗留问题细节 —— 是否入库由用户决定（建议入库，它是最完整的交接文档）

## 6. 发布前命令清单（由用户执行，本交接未执行）

```bash
git status --short | grep -E "hko_api/data/raw|data/|outputs/" || echo "OK: 大文件未入库"
node hko_api/district/tests/bulletin.test.mjs    # 8 passed
python -m pytest                                  # 167 passed
# 人工 375px 复核 → 首次 commit → LICENSE → remote → push（均需用户批准）
```
