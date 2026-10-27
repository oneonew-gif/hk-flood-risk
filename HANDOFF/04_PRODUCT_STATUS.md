# 04 · 产品轨道现状

> 核查日期：2026-10-27。产品轨道全部产物位于 `hko_api/` 独立目录，与论文冻结数据完全隔离。

---

## 1. HKO 实时接入（`hko_api/hko_fetch.py`）— 已完成 ✅

- **五源**（HKO Open Data，`data.weather.gov.hk`，**无需 API 密钥**，纯标准库实现）：

| 源 | 内容 | 轮询 | STALE 阈值 |
|---|---|---|---|
| warnings | 生效天气警告（暴雨黄/红/黑、雷暴、台风等） | 60s | 15 min |
| rhrread | 18 区过去 1 小时最大雨量 | 300s | 30 min |
| hourlyRainfall | 自动气象站过去 1 小时雨量 | 300s | 90 min |
| hhot | 鲗鱼涌(QUB)逐时天文潮（调和**预报**，非实测） | 3600s | 26 h |
| nowcast | 未来 2h 网格降雨临近预报（HK bbox 裁剪后 ~900 格 × 4 步，~2km 分辨率） | 720s | 30 min |

- **存储**：原始响应 `data/raw/{source}/{YYYYMMDD}/{source}_{HHMMSS}.{json|csv}` + 同名 `.meta.json`；日志 `data/raw/fetch_log.jsonl`；归一化快照 `data/latest/snapshot.json|js`
- **降级**：网络失败保留该源上次成功快照 + `stale:true` + `last_error`，不崩溃、不伪装实时
- **修正 A 方清单 3 处口径**：HHOT 是天文潮预报非实测；nowcast ~2km 非 1km；nowcast 数据集 ID 已更正
- 运行：`.venv/bin/python -m hko_api.hko_fetch --loop`（`--once` 单次；`--only warnings,rhrread` 部分源）

## 2. 网格评分准备（`hko_api/prepare_scores.py`）— 已完成 ✅

- 只读输入：冻结 `predictions_test.parquet` + `gis/grid_risk_500m.csv`
- 输出 `data/latest/grid_scores.json|js`：5,517 格，测试期均分 min 0.0001 / med 0.0049 / max 0.0990；双口径（测试期 2023-01~2026-09 / 全期 train+val+test）分别标注
- 几何诚实：仅有格中心点，**未伪装网格边界**

## 3. 整合 Dashboard（`hko_api/dashboard.html`）— 已完成 ✅（8/8 E2E 通过）

- Leaflet 1.9.4 已 vendor 到 `hko_api/vendor/`，离线/file:// 双击可用
- **三类图层显式区分**：观测（警告/各区雨量/站点雨量）｜官方预报（nowcast 叠加层、天文潮曲线）｜历史模型评分（地图圆点，弹窗含「弱监督标签下探索性风险排序，非实时水浸概率」披露）
- 测试期/全期双底图切换、五分位色阶、单 canvas 渲染器、逐源新鲜度表、STALE 横条
- 页头常驻免责：「实时气象 = HKO 官方数据 ｜ 地图评分 = 历史模型输出（弱监督探索性）」
- 访问：`http://127.0.0.1:8321/hko_api/dashboard.html`（服务须从**项目根目录**启动）

## 4. 18 区气象播报（`hko_api/district/`）— 已完成 ✅（T6）

- `index.html`：18 区选择器（localStorage 记忆）、过去 1h 雨量播报卡片、全港级警告卡片、数据新鲜度表、历史风险独立入口（`../dashboard.html`）
- `bulletin.js`：纯函数规则模板（雨量 6 级分档为本产品展示规则、非天文台官方分级；缺测/过期/失败分支不编造数值）
- 测试：`node hko_api/district/tests/bulletin.test.mjs` → **8 passed / 0 failed**（2026-10-27）
- 访问：`http://127.0.0.1:8321/hko_api/district/`
- 限制：无 18 区边界几何（仅列表，不做分区填色）；nowcast 无分区产品；`dc` 编码无区名映射（不做风险按区聚合）

## 5. STPU 区域分析底表（T3，`outputs/t3_stpu_base_20261027/`）— 底表已完成，正式分析被 P0 阻断 ⚠️

- `stpu_base_table.csv`（211 STPU × 20 字段）+ 字段词典 + 数据来源表 + 质量报告 + `open_issues.md`（13 项）
- 聚合链路已验证：Σ(格评分×overlap_frac)/Σoverlap_frac 独立复现最大差 4.95e-07；**必须用 4 位小数长表**
- 质量审计（`outputs/t3_quality_audit_20261027/`）：17 项 = 通过 11 / 待验证 4 / 不适合正式使用 2（SVI 不可复算 R²=0.647；exposure_population ≡ pop_total 重复计数陷阱）
- **P0 阻断项**：① STPU 边界几何缺失 ② 2021 普查原始底表未交付 ③ SVI 方法未公开 ④ 46/207 STPU 人口与 WorldPop 比值超 [0.5,2]（键错配风险）→ **不生成区域排名、不生成高风险×高脆弱交集**

## 6. 运行状态与操作（迁移必读）

**当前运行中**（旧机器，2026-10-27 14:40 实测）：

| 进程 | 端口/输出 | 停止 | 重启 |
|---|---|---|---|
| `hko_fetch --loop`（PID 69070） | 写 `hko_api/data/` | `kill 69070` | 项目根目录 `.venv/bin/python -m hko_api.hko_fetch --loop` |
| `http.server 8321`（PID 66215） | http://127.0.0.1:8321 | `kill 66215` | 项目根目录 `.venv/bin/python -m http.server 8321 --bind 127.0.0.1` |
| `http.server 8322`（PID 49015） | /tmp/t6_fixtures 测试夹具 | `kill 49015` | 一般无需重启 |

> 迁移后在新机器重启前，确认旧机器进程已停（避免双方同时写归档/占端口）。新机器首次运行 `hko_fetch --loop` 会从零开始积累归档，属正常。

## 7. 已知待解决问题（产品）

1. **16 个再分析降雨特征无法实时原口径获取**（POWER/ERA5 延迟数天）；hourlyRainfall 仅最新 1h 快照，prev24h/48h/api72 需 72h 轮询冷启动；换 HKO 源存在再分析 vs 站点观测的 distribution shift（须并行记录两路输入监控偏差）
2. **实时 XGBoost 推理未实施、未批准**：59 特征中 37 静态可直接复用、4 直接可得、18 需聚合（含 16 雨特征）；接入前须独立版本 + 特征兼容性验证 + 用户批准；**不得展示「实时水浸概率」**
3. P1 未做：网格-观测联动（点网格高亮最近雨量站）、演示模式（脚本化播放）
4. Dashboard 验收后期截图工具不可用，部分验收以 DOM 断言为证；移动端 375px 未真机实测
5. nowcast 全量原始落盘约占磁盘（raw 280 MB 且持续增长），如需限制可改仅保留快照（需决策）
6. `model_demo.html`（根目录 245 KB）：早期单文件交互演示，独立产物，与 dashboard.html 不同；维护状态待确认
