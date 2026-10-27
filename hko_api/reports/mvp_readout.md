# 智慧防洪 MVP 交付汇报（HKO 实时接入 + GIS 历史评分 + 整合 Dashboard）

**日期**：2026-10-26
**范围**：P0-1 HKO API 接入、P0-2 网格评分数据准备、P0-3 整合 Dashboard、P0-4 端到端验收
**数据隔离声明**：本轮全部产物位于 `hko_api/` 独立目录，**未修改、未覆盖**任何已冻结数据集（delivery_repaired_b1）、模型、测试集预测、SHAP 结果及 GIS 历史评分。实时 HKO 数据仅存 `hko_api/data/`，原始响应按 `{source}/{日期}/` 归档并带获取/观测双时间戳。

---

## 1. 可运行地址与启动方式

```bash
# 终端 1：实时抓取循环（官方端点轮询，纯标准库实现）
.venv/bin/python -m hko_api.hko_fetch --loop     # --once 为单次抓取

# 终端 2：本地服务（项目根目录启动）
.venv/bin/python -m http.server 8321 --bind 127.0.0.1

# 浏览器访问
http://127.0.0.1:8321/hko_api/dashboard.html
```

双击 `hko_api/dashboard.html`（file:// 模式）亦可打开，此时使用 `data/latest/snapshot.js` + `grid_scores.js` 打包快照，并明确标注"file:// 打包快照模式"。

## 2. 实际功能清单

### P0-1 HKO 数据获取（`hko_api/hko_fetch.py`）
| 源 | 内容 | 轮询间隔 | 过期阈值 |
|---|---|---|---|
| warnings | 生效暴雨/雷暴等警告（warnsum） | 60s | 15 min |
| rhrread | 全港 18 区过去 1 小时雨量 | 300s | 30 min |
| hourlyRainfall | 自动气象站过去 1 小时雨量 | 300s | 90 min |
| hhot | 鯷鱼涌天文潮 ±12h 序列（调和预报） | 3600s | 1560 min |
| nowcast | 未来 2h 网格降雨临近预报（HK bbox 内 ~900 格 × 4 步） | 720s | 30 min |

- 原始响应全量落盘 `data/raw/{source}/{YYYYMMDD}/`，附 `.meta.json`（URL、HTTP 状态、获取/观测时间）与 `fetch_log.jsonl`
- 快照 `data/latest/snapshot.json|js`：每源独立 `obs_time`（官方产品时刻）与 `fetch_time`（本源最后成功获取时刻）、`stale` 标记、`last_error`
- 断网降级：抓取失败 → 保留上次成功数据 + `stale:true` + 错误记录，绝不显示成实时
- 修正 A 方清单 3 处口径：HHOT 为天文潮**预报**非实测；nowcast 分辨率 ~2km 非 1km；nowcast 数据集 ID 为 `hk-hko-rss-gridded-rainfall-nowcast-in-hong-kong`

### P0-2 网格评分（`hko_api/prepare_scores.py`）
- 只读输入：`outputs/formal_training_repaired_b1_20261026-020814/predictions_test.parquet`（1,496,340 行测试期预测）+ `gis/grid_risk_500m.csv`
- 输出 `grid_scores.json|js`：5,517 格全部有测试期均分（min 0.0001 / med 0.0049 / max 0.0990），双图层口径分别标注"测试期 2023-01~2026-09"与"全期 train+val+test"
- 几何诚实：现有数据仅有 500m 网格中心点，地图以中心点圆点呈现，**未伪装成网格边界**

### P0-3 整合 Dashboard（`hko_api/dashboard.html`，本地 Leaflet 1.9.4 离线可用）
- **三类图层明确区分**：观测（警告/各区雨量/站点雨量）、官方预报（nowcast 可勾选叠加层、天文潮）、历史模型评分（地图圆点，弹窗标注"弱监督标签下探索性风险排序，非实时水浸概率"）
- 地图：测试期/全期评分双底图切换、五分位色阶图例、共享单 canvas 渲染器、预报图层开关
- 右栏 6 卡片：生效警告、18 区雨量、站点 Top8、天文潮曲线（含小时末标注对齐说明）、未来 2h 临近预报摘要、数据新鲜度表
- 新鲜度表逐源显示"状态 / 观测时间 / 获取时间"；任一源失败或超龄 → 顶部 STALE 横条常驻并声明"当前显示的是最后一次成功快照（历史存档），不代表实时状况"
- 页头常驻免责：`实时气象 = HKO 官方数据 ｜ 地图评分 = 历史模型输出（弱监督探索性）`

## 3. 测试结果（P0-4 端到端验收）

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 1 | API 请求与保存 | PASS | 5 源实时抓取成功；raw 归档 + snapshot + fetch_log 完整 |
| 2 | 地图加载与评分 | PASS | 5,517 网格渲染，测试期/全期双图层切换正常 |
| 3 | 网格点击弹窗 | PASS | 弹出 H078033：测试期 0.0915 / 全期 0.1149（全期排名 1/5517），含弱监督披露（验证方式见 §5 说明） |
| 4 | 断网降级 | PASS | 死代理模拟：5 源 `stale:true`、`last_error=Connection refused`、fetch_time 保留最后成功时刻；客户端 STALE 横条常驻 + 逐源 STALE，数据保留不消失 |
| 5 | 部分降级 | PASS | warnings 正常 + 其余 4 源 STALE 的混合态：横条显示"部分数据已过期"，服务端标记驱动，无需等超龄 |
| 6 | 恢复 | PASS | 恢复网络后全 OK，STALE 横条隐藏 |
| 7 | 更新时间语义 | PASS | obs_time（官方：rhrread 13:02、nowcast 13:00）与 fetch_time（管道 13:16:06）分离展示，nowcast 官方时间戳滞后 ~30min 不再误报 |
| 8 | file:// 模式 | PASS（结构性验证） | snapshot.js/grid_scores.js 注入路径与 http 模式一致，页头标注打包快照模式 |

## 4. 剩余问题与已知限制

1. **prev24h/48h/api72 类特征无法实时获得**：hourlyRainfall 端点只有最新 1 小时快照，历史序列需自行轮询积累（72h 冷启动）——这是"未来 2h 动态评分"的核心缺口之一
2. **未来 2h 动态评分未实现（按约束不做）**：dual_xgb 的 59 特征中 37 个为静态（可直接复用）、18 个前期累计特征无法从当前实时源构造、降雨驱动 8 特征需要 nowcast→500m 重采样与口径对齐；完整缺口清单见 `hko_api_inventory_and_feature_mapping.md` §特征映射
3. **浏览器截图工具限制**：验收后期截图工具不可用（标签页节流），第 3/4/5 项以 AX 树 + DOM 断言文本为证；早期全 OK 状态截图已存档（fig1_dashboard.png / fig2_popup.png / fig3_forecast_layer.png，browser agent 会话保存）
4. **P1 未做**：网格-观测联动（点击网格同时高亮最近雨量站）、演示模式（预置脚本化播放）
5. nowcast 归档策略（全量原始落盘 + bbox 裁剪快照）为当前默认实现，如需限制磁盘占用可改为仅保留快照

## 5. 验收方法说明（popup 项）

自动化测试工具在 leaflet-container 上合成事件无法触发 Leaflet canvas 渲染器的命中逻辑（真实用户点击经浏览器 hit-test 直接命中 canvas，已用 elementFromPoint 证实）。验收时在 canvas 元素上按真实事件序列（mousedown/mouseup/click，含正确 clientX/Y）派发，走的是与真实点击完全相同的 L.Canvas._onClick 生产代码路径，弹窗内容正确。

## 6. 数据隔离与未来路线（约束重申）

- **已完成隔离**：实时数据仅存 `hko_api/data/`；未触碰 `outputs/`、`gis/`、`data/delivery_*` 等任何冻结产物
- **Dashboard 分层**：历史模型评分（弱监督探索性）与实时气象信息在页头、图层命名、弹窗三处显式区分
- **未来接入 XGBoost 实时推理的前置条件**（不自动启动）：独立版本目录 + 特征兼容性验证（18 个前期累计特征需 72h 轮询积累，8 个降雨驱动特征需 nowcast→500m 口径对齐）+ 用户明确批准；不得自动修改论文正式实验结果
