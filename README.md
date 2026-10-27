# 水浸风险预测建模框架（XGBoost + GIS）· 模块 B

对接筑工杯「智慧防洪」项目的 B 部分：模型训练、模型评价、风险概率预测、三级风险划分、GIS 结果导出。

**当前进度：批 1（数据层 + 切分 + 泄漏闸门，已按"预测发起时刻"口径重构）+ 批 2（特征 / 训练 / 调参 / 评价 / 校准 / 端到端流水线）已完成。批 3–4 未开始。**

---

## 一、这套框架的三条硬规矩

在跑任何代码之前请先读这一段，因为它决定了你答辩时能不能自圆其说。

### 1. 三个决定必须由人做，框架拒绝替你猜

| 决定 | 为什么不能交给程序 | 框架怎么处理 |
|---|---|---|
| 标签口径 | `flood_label=1` 是"历史上发生过"还是"某时段内发生"，直接改变模型含义 | 只校验取值必须是 0/1，口径由你在数据侧定义，`io` 不做任何推断 |
| 训练/测试切分 | 按行随机切会让相邻网格同时进训练和测试，AUC 虚高十几个点 | 提供 `temporal` / `spatial_block` / `spacetime` 三种防泄漏切分；没有坐标时**报错而不是退化成随机** |
| 三级风险阈值 | 0.33/0.66 是拍脑袋，与历史命中率、预警成本无关 | `tiering.strategy=fixed` 时若不给 `cutoffs` **直接报错退出**，代码里没有任何默认阈值 |

### 2. 合成数据有红线

`data/synthetic/` 下所有数据由 `floodrisk.synthetic` 生成：

- 坐标原点在 **(0°, 0°)** 附近的几内亚湾，与香港无任何几何关系；
- 每个数据文件旁挂 `*.provenance.json`，`io` 读取后把 `SYNTHETIC` 标记带进 dataset、指标与 manifest；
- **其任何指标、概率、地图都不得写进参赛报告或答辩材料**。它的唯一用途是证明"程序能跑通"。

### 3. 人口/脆弱性默认不进模型

`population`、`vulnerability_score` 默认 `features.vulnerability.in_model: false`。

理由：人口密度不是水浸的**物理成因**。把它喂给 XGBoost，特征重要性会把人口推到前列，而答辩时"人口多是水浸的原因吗"这个问题无法回答。正确做法是让它走**独立的脆弱性叠加**（`vulnerability.py`，批 3），产出"高风险 × 高脆弱"优先社区清单 —— 这才是治理章要的材料。

要纳入模型可以，改一个开关即可，但 `manifest` 会记录 `vulnerability_in_model: true`，便于你主动交代。

---

## 二、30 秒跑通（合成数据）

```bash
# 1. 依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # 可选：以可编辑包方式安装

# 2. 生成合成数据（会同时写 *.provenance.json 与 grids_synthetic.geojson）
PYTHONPATH=src python -c "
from floodrisk.synthetic import write_synthetic_bundle
print(write_synthetic_bundle('data/synthetic', mode='dynamic', n_grids=64, n_steps=240, seed=42))"

# 3. 测试
python -m pytest

# 4. 跑完整训练流水线（批 2）
PYTHONPATH=src python -c "
from floodrisk.config import Config
from floodrisk.pipeline import run_training
m = run_training(Config.load('configs/synthetic_dynamic.yaml'))
print(m['metrics']['test']['pr_auc'])"
```

> **macOS 上 XGBoost 需要 libomp。** 若 `import xgboost` 报
> `Library not loaded: @rpath/libomp.dylib`，装一个 OpenMP 运行时即可
> （`brew install libomp`；本机因为 brew 锁目录受系统限制，改为从官方 bottle
> 里取出 `lib/libomp.dylib` 放到 `/opt/homebrew/opt/libomp/lib/`）。
> 这一步与项目代码无关，但换机器时一定会踩到。

### 批 1 的冒烟验证脚本

```python
from floodrisk.config import Config
from floodrisk.io import load_dataset
from floodrisk.splits import make_split_plan
from floodrisk.leakage import audit_features

cfg = Config.load("configs/synthetic_dynamic.yaml")
ds  = load_dataset(cfg)                 # 校验报告在 ds.report
plan = make_split_plan(ds, cfg)         # train/val/test 行索引
audit = audit_features(ds, cfg)         # 三道闸 + 单变量 AUC 扫描

print(ds.report["stats"], ds.report["label"])
print(plan.summary_with_labels(ds.y))
print(plan.meta["time_boundaries"], plan.meta["purged_rows"])
print(audit["suspect_features"])
```

---

## 三、目录结构

```
hk-flood-risk/
├─ configs/
│  ├─ base.yaml                 # 所有默认值的唯一来源
│  ├─ synthetic_dynamic.yaml    # 动态模式 · 合成数据
│  └─ synthetic_static.yaml     # 静态模式 · 合成数据
├─ src/floodrisk/
│  ├─ config.py     ✅ 配置加载/校验/派生量/指纹
│  ├─ schema.py     ✅ 字段契约（规范名 ↔ 别名，含预报类）
│  ├─ io.py         ✅ 读入 + 列映射 + 校验报告
│  ├─ synthetic.py  ✅ 合成数据（仅自测，含缺报注入）
│  ├─ splits.py     ✅ 四种切分 + 标签窗口 purge + 同源 CV
│  ├─ leakage.py    ✅ 三道泄漏闸门（按时间位移）+ 单变量 AUC
│  ├─ features.py   ✅ 批 2：按时间轴 lag/rolling、预报窗口聚合、静态直通
│  ├─ train.py      ✅ 批 2：XGB 训练、早停、不平衡、重要性
│  ├─ tuning.py     ✅ 批 2：随机搜索（折与主切分同源，含 purge）
│  ├─ evaluate.py   ✅ 批 2：7 指标 + 混淆矩阵 + 阈值扫描 + 曲线
│  ├─ calibrate.py  ✅ 批 2：none / platt / isotonic + 可靠性曲线 + ECE
│  ├─ pipeline.py   ✅ 批 2：端到端串联 + manifest + 诊断告警
│  ├─ tiering.py    ⬜ 批 3：三级划分
│  ├─ vulnerability.py ⬜ 批 3：独立叠加
│  ├─ predict.py    ⬜ 批 3：网格级分数
│  ├─ gis_export.py ⬜ 批 4：grid_id 关联 → CSV/GeoJSON
│  └─ cli.py        ⬜ 批 4：一条命令跑通
├─ tests/           ✅ 125 项
└─ outputs/         # 每次运行的快照、指标、预测、地图、manifest
```

---

## 四、换成 A 的真实数据：改三处

不需要动代码，只改 `configs/` 里一份自己的配置（建议 `configs/real_dynamic.yaml`，`extends: base.yaml`）。

**① 数据路径**

```yaml
data:
  input_path: /path/to/A_delivered_panel.csv
  format: csv                     # csv | parquet | xlsx
  grid_layer_path: /path/to/grids.geojson   # 没有几何就留空
  strict: true                    # 真实数据阶段建议开：校验失败就停，不要带伤跑
```

**② 列名映射**（只写和数据列名不一致的字段）

```yaml
data:
  columns:
    grid_id: GRID_ID
    timestamp: OBS_TIME
    rainfall: RF_1h
    tide: tide_mCD
    elevation: DEM_m
    flood_label: FLOODED
```

字段别名已内置常见写法（`rain_mm`、`precip`、`tide_level`、`DEM`、`is_flood`…），先跑一遍看 `ds.report["unresolved_fields"]`，缺什么再补什么。

**③ 时间戳约定 + 任务类型 + 切分**

```yaml
data:
  # 只有 A 能回答：整点雨量是"过去一小时累计"还是"未来一小时预告"？
  time_conventions:
    rainfall:  {aggregate: interval, anchor: end, reporting_delay_steps: 0}
    tide:      {aggregate: instantaneous, reporting_delay_steps: 0}
task:
  mode: dynamic
  anchor: issue_time       # 行时间戳 = 预测发起时刻
  horizon_hours: 2         # 只定义标签窗口 [t, t+2h)
split:
  strategy: spacetime      # 正式结果建议用最严格的；另附 temporal 作对照
  spatial_block_column: district
  purge_horizon: true
```

### 换数据后第一件事：读校验报告

```python
ds.report["issues"]            # 每条都有 level / code / message
ds.report["features_dropped"]  # 每个被剔除的特征都写了原因
ds.report["unresolved_fields"] # 哪些字段没匹配上
ds.report["per_column"]        # 缺失率、越界数、单位、min/max
```

**框架不填缺失值、不改越界值。** 它们只出现在报告里，怎么处理由你决定 —— 这是有意的。

---

## 五、切分策略怎么选

| 策略 | 含义 | 什么时候用 | 已知偏差 |
|---|---|---|---|
| `random` | 按行随机 | 只用于程序冒烟 | 同时泄漏空间与时间，指标不可引用 |
| `temporal` | 同址、未来时段 | 贴近业务运行（用过去预测未来） | 同一批地点跨集合，指标偏乐观 |
| `spatial_block` | 未见空间块 | 检验对未见地点的泛化 | 不检验时间外推 |
| `spacetime` | 未见块 + 未来时段 | **正式报告主结果** | 最保守，样本损失最多 |

`temporal` 会在 `plan.meta["blocks_shared_with_test"]` 里如实记录训练集与测试集共享了多少个空间块 —— 写报告时把这句话写进去，比被评委问出来好。

### 标签窗口要 purge，否则两段共享同一场雨

标签是 `[t, t+horizon)` 的窗口标签。如果训练集最后一行是 `T-1`、验证集从 `T` 开始，那么训练行的标签窗口就伸进了验证段 —— 两个集合看到的是**同一批水浸事件**，评估会系统性偏乐观。

`split.purge_horizon: true`（默认）会在 train/val、val/test 之间剥掉宽度为 horizon 的清洗带，CV 折之间同样处理。代价是每段末尾损失几小时样本，`plan.meta["purged_rows"]` 会把剥掉多少行交代清楚。

---

## 六、时间口径：三个必须分开的东西

这一段是重构的核心。**预测跨度是标签侧的参数，不是特征侧的参数**，混在一起会同时导致"丢掉最有价值的近期雨量"和"以为安全其实不安全"。

| 概念 | 定义 | 由谁决定 |
|---|---|---|
| 预测发起时刻 `t` | 行的时间戳。此刻你只能用手上已有的数据 | `task.anchor: issue_time` |
| 标签窗口 `[t, t+h)` | `flood_label=1` 当且仅当窗口内发生水浸 | `task.horizon_hours` |
| 特征可用性 | 某个变量在 `t` 时刻是否**已完整观测** | `data.time_conventions`（聚合方式 + 时间戳锚 + 上报延迟） |

举例：12:00 发起，预测 12:00–15:00。`[10:00, 11:00)` 的雨量在 12:00 早已观测完毕，**可以用，不是未来信息**。只有"11:00 之后才发生"的观测才算越界。

### 两种锚定口径

- **`issue_time`（默认，业务口径）**：行时间戳 = 发起时刻。特征只要在 `t` 已观测完即可用，最小滞后由时间戳约定推出，**与 horizon 无关**。三级预警必须用这个口径，否则产不出触发条件。
- **`window_end`（复盘口径）**：行时间戳 = 标签窗口末端。特征必须整段早于窗口打开，最小滞后 = `horizon_steps` + 上报延迟。用于事后归因，不适合预警。

### 时间戳约定决定"lag 多少才安全"

| 变量约定 | 含义 | 最小合法滞后（issue_time 口径） |
|---|---|---|
| `interval` + `anchor: end` | 整点值 = 过去一小时累计 | `reporting_delay_steps` |
| `interval` + `anchor: start` | 整点值 = 之后一小时累计 | `1 + reporting_delay_steps` |
| `instantaneous` | 瞬时读数 | `reporting_delay_steps` |
| 预报类（天文潮） | 未来值但在 `t` 完全可知 | 不看滞后，看窗口聚合 |

**A 没确认约定之前，`features.temporal.min_lag_policy: conservative` 强制 lag ≥ 1**（代价：放弃最近一小时雨量；收益：杜绝"整点值是时段起点还是终点"读反导致的整批泄漏）。A 书面确认后改成 `derived`，由约定自动推出，`end` 锚定 + 0 延迟时 lag 0 自动放开。

### 预报类特征（天文潮）

天文潮是确定性过程：`[t, t+2h)` 的潮位在 `t` 时刻可以精确算出。它**既是未来值，又毫无泄漏**。`features.forecast` 就是为这类变量开的通道 —— 单点值意义有限，真正有信息量的是未来窗口上的最大潮位，批 2 做窗口聚合。

NWP 预报雨同理可用，但必须带**发行时刻戳**，回测时只能用 `t` 之前发行的预报。A 交付前不开。

---

## 七、泄漏防线（三道闸 + 一道扫描）

| 闸门 | 位置 | 行为 |
|---|---|---|
| 特征黑名单 | `check_feature_names` | 命中 `forbidden_patterns`（水浸深度/报告数/事后清淤/应急响应）**直接抛错**，不给 try 包一层绕过的机会 |
| 时序对齐 | `check_time_lag` / `check_time_rolling` | **按时间轴**校验，不是按行：先把每个网格重索引到完整时间轴，再验证 `feature[t] == base[t-lag·step]` |
| 切分不相交 | `assert_split_disjoint` | 行级 + 空间块级双重检查 |
| 单变量 AUC | `univariate_auc_scan` | 单个特征 AUC > 0.97 就标红。只告警不中止 —— 需要你判断该变量的可观测时点 |

时序闸门为什么必须按时间轴：面板一旦有缺报（某网格某小时没数据），`df.shift(3)` 是**按行**位移，实际对齐到的可能是 4 小时前。更阴险的是，跨缺口后的第一段恰好落在 NaN 上，等值比较会跳过它、看起来"没问题"。所以校验分两步：

1. 期望值可得 → 必须严格相等；
2. 期望值不可得（该时点在时间轴上根本没观测）**但特征有值** → 只能是按行位移跨过了缺口，报错。

合成数据专门注入了 1.5% 的缺报行 —— 没有缺口，"按行"和"按时间"永远一致，这套校验就是摆设。

---

## 八、训练与评价（批 2）

一次 `run_training` 会在 `outputs/<时间戳>_<配置指纹>/` 下产出：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 配置指纹、数据来源、样本损失、切分、指标、校准、诊断告警 —— 答辩追溯就靠它 |
| `model.json` | XGBoost 模型本体 |
| `calibrator.json` | 校准器参数（platt 系数 / isotonic 拐点），可脱离代码复现 |
| `predictions_val_test.csv` | 逐行预测：`grid_id` / `timestamp` / `y_true` / `prob_raw` / `prob_calibrated` |
| `threshold_sweep_val.csv` | 验证集逐阈值 P/R/F1 |
| `feature_importance.csv` + `_permutation.csv` | gain/weight/cover 与验证集置换重要性 |
| `curves_test.json` | ROC / PR 曲线坐标（批 4 出图直接用） |
| `config_snapshot.yaml` / `validation_report.json` | 当日配置与数据体检报告 |

### 合成数据上跑出来的数（`SYNTHETIC_DO_NOT_CITE`，仅为验收程序）

| split | n | 正例 | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| train | 7088 | 972 | 0.935 | 0.743 | 0.461 | 0.853 | 0.599 |
| val | 2411 | 165 | 0.693 | 0.205 | 0.236 | 0.333 | 0.276 |
| test | 2534 | 384 | 0.713 | 0.315 | 0.272 | 0.469 | 0.345 |

PR-AUC 基线（正例率）= 0.152，所以测试集是 **2.1 倍提升**。这个数字不好看，但它是对的：
特征只有滞后观测，而标签看的是未来 2 小时 —— 模型看不到把雨带过来的那场风暴。
> 注意：上表数字产生于 **3 小时窗口**的历史运行（2026-10-22），改为 2h 后未重跑，
> 与当前配置不一一对应，只用于说明"程序跑通"，不可作为研究结论。
**真数据上大概率也不会好看**，这本身就是可写进报告的发现，不是代码 bug。

### 三个必须主动交代的问题（框架自己会写进 `manifest["warnings"]`）

1. **训练/测试落差大**（0.743 vs 0.315）。时间切分下模型记住了训练期的特定事件。
   只贴训练集指标是这类项目最典型的自欺，所以流水线会把落差直接算出来告警。
2. **加权但未校准**。`scale_pos_weight` 把分数系统性推高（测试集平均预测 0.35，
   实际发生率 0.15）。此时分数**不能解释为概率**，也不能用绝对阈值分级。
3. **基准率漂移**。三段正例率 13.7% / 6.8% / 15.2%，验证段恰好偏干，
   导致在验证集上拟合的校准器到测试段整体偏高。→ 三级分级应优先用**分位数**策略。

校准确实有效（isotonic，测试集）：**Brier 0.181 → 0.122，ECE 0.242 → 0.060**。
但注意 ECE 降下来了，平均预测值仍偏离实际发生率 —— 校准能修"形状"，修不了"基准率漂移"。

### 特征重要性：gain 与置换重要性要一起看

合成数据上的 gain 前五是 `elevation` / `rainfall_lag1` / `wind_speed_lag1` /
`tide_ast_fw3max` / `rainfall_lag3`；置换重要性前五是 `elevation` / `rainfall_lag1` /
`drainage_capacity` / `slope` / `tide_ast_fw3max`。

差异不是 bug：gain 是训练集内的分裂收益（偏爱高基数、连续型特征），
置换重要性是在验证集上测"打乱这一列后 PR-AUC 掉多少"。
**写报告时用置换重要性讲贡献，用 gain 讲模型结构**，别混着说。

### 超参搜索为什么不用 `RandomizedSearchCV`

它的默认 CV 是 KFold —— 同一空间块、相邻时段会被切到两侧，三道闸全部失效。
所以 `tuning.py` 自己写循环，折由 `splits.cv_splits` 提供（与主切分同源，含 purge 与空间分组）。
另外它返回的是**全表行位置**，和 `RandomizedSearchCV` 期望的 `0..n-1` 坐标系不同，
硬塞进去会静默错位 —— 这也是不用它的原因之一。

---

## 九、测试

```bash
python -m pytest              # 125 passed
python -m pytest -k leakage   # 只跑泄漏闸门
```

测试覆盖的重点不是"函数能跑"，而是这些**必须失败**的场景真的会失败：

- `fixed` 分级策略缺 `cutoffs` → 报错
- 静态模式配了时序 lag → 报错
- 特征名命中 `flood_report_count` → 报错
- 只声明 `lag=2` 实际按行位移跨了缺口 → 数值校验抓出
- 滚动窗口跨缺口拼数据 → 抓出
- 把实测雨量塞进预报窗口 → 报错
- 滞后特征被写反成 lag1 → 复核抓出
- 打乱标签后调参得分仍然很高 → 说明打分用错了数据
- 没有坐标时做空间切分 → 报错，不退化成随机
- 切分行重叠 / 空间块共享 → 报错
- 标签窗口跨段重叠 → purge 后断言不成立就失败

---

## 十、已知限制（写报告时可以直接用）

1. **合成数据的标签率是人为设定的**，不是从现实学到的。真实正例率由历史水浸记录决定，通常远低于此。
2. **时间戳约定尚未由 A 书面确认**。当前按天文台惯例假设（整点值 = 过去一小时累计），并用 `conservative` 策略兜底；若实际是起点锚定，`derived` 模式下 lag 0 会有 1 小时泄漏风险，需重跑。
3. `temporal` 切分无法排除同一地点的历史信息，指标对"新地点"的泛化是乐观的。
4. 空间分块的 `KMeans` 版本只按质心聚类，不保证块内连通性（行政区版没有这个问题）。
5. 斜坡单元/网格边界效应未处理；`buffer_meters` 目前只是配置占位，尚未实现剔除。
6. 缺 `drainage_capacity` 这类规划参数时，模型会退化成"地形 + 气象"驱动，解释力下降 —— 但这属于数据可得性问题，不该用代理变量硬凑。
7. 预报类特征目前只接入了天文潮；NWP 预报雨需要发行时刻戳，等 A 交付后再评估。

---

## 十一、协作者首次启动（HKO 实时模块）

> 本节写给 clone 本仓库的人：**不需要任何私钥、不需要作者本机的任何文件或后台进程**，
> 只需 Python 3 与（可选）Node。建模框架本体的启动见上文第二节。

### 0. 环境

```bash
git clone <仓库地址> && cd hk-flood-risk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 若需跑 prepare_scores.py / 读 parquet，额外装：pip install pyarrow
```

### 1. 抓取香港天文台实时数据（生成 `hko_api/data/latest/`）

```bash
python3 hko_api/hko_fetch.py --once     # 抓一次即退出
# python3 hko_api/hko_fetch.py --loop   # 常驻轮询（可选，非启动必需）
```

- 纯标准库实现，**无第三方依赖、无密钥**（HKO 为开放数据）；
- 所有输出路径由 `Path(__file__).resolve().parent` 推导，**不含任何作者本机绝对路径**；
- 产物写到 `hko_api/data/latest/`（`snapshot.json` + `snapshot.js`），
  该目录与 `hko_api/data/raw/` 均**不入库**，因此 clone 后必须先跑这一步。

### 2. 打开 18 区气象播报页

```bash
python3 -m http.server 8000             # 必须在仓库根目录起（页面用 ../data/latest/ 相对路径）
# 浏览器打开 http://localhost:8000/hko_api/district/
```

页面本身是静态的，**不需要后端**；起 http.server 只为绕开浏览器对 `file://` 的 fetch 限制。

### 3. 规则单测（可选，需 Node）

```bash
node hko_api/district/tests/bulletin.test.mjs    # 预期 8 passed
```

### ⚠️ 唯一无法自行生成的产物：历史风险地图图层

`hko_api/dashboard.html` 的网格风险图层依赖 `hko_api/data/latest/grid_scores.json`，
它由 `hko_api/prepare_scores.py` 从
`outputs/formal_training_repaired_b1_20261026-020814/` 只读转换而来。
**该 `outputs/` 目录不入库**（体积与数据授权原因），因此：

- 协作者能自行跑通：实时气象播报（第 1–2 步）✅
- 协作者**无法**自行生成：历史风险评分图层 ❌ —— 需要作者另行提供该文件，
  或提供 `outputs/` 下对应的正式训练产物（`predictions_test.parquet` + `gis/grid_risk_500m.csv`）。

注意：`prepare_scores.py` 只做只读转换，不调用模型、不改动任何正式实验产物。

### 数据授权提醒

HKO 数据知识产权属香港特区政府。非商业用途须接受其《网站资料的使用条件（非商业用途）》
（须注明政府为知识产权拥有人、注明资料来源网站、转载其免责声明）；商业用途须事先书面授权。
本仓库为**私有仓库**阶段仅内部使用；若日后转公开，需在 README 与页面显著位置补齐上述署名与免责声明。
