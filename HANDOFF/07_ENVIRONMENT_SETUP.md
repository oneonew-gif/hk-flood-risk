# 07 · 环境配置与运行

> 实测环境：macOS 26.7.1 arm64（Apple Silicon），2026-10-27 核查。

---

## 1. 运行时版本（`.venv/` 实测）

| 组件 | 版本 |
|---|---|
| Python | **3.12.14**（`requires-python >= 3.10`） |
| pandas | 3.1.0（**注意：pandas 3.x**，与 2.x 有行为差异，复现时用 uv.lock 锁定） |
| numpy | 2.5.4 |
| xgboost | 3.4.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| pyarrow | 26.0.0 |
| joblib | 1.6.0 |
| matplotlib | 3.11.2 |
| PyYAML / pytest | ≥6.0 / ≥8.0（pytest 实测 9.1.1） |
| Node | 任意现代版本（仅用于 `district/tests/bulletin.test.mjs`，无 npm 依赖） |

## 2. 新机器安装

```bash
cd hk-flood-risk

# 方式 A：uv（推荐，uv.lock 已锁定全部依赖）
uv sync   # 或 uv pip install -r requirements.txt

# 方式 B：venv + pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # 可编辑安装 src/floodrisk
```

**macOS 必踩坑 — XGBoost 需要 libomp**（README §二）：
- `import xgboost` 报 `Library not loaded: @rpath/libomp.dylib` → `brew install libomp`
- 若 brew 锁目录受限：从官方 bottle 取 `lib/libomp.dylib` 放到 `/opt/homebrew/opt/libomp/lib/`

## 3. 环境变量与密钥

- **本项目无 API 密钥**：HKO Open Data 无需认证；`hko_fetch.py` 纯 urllib 标准库
- 仓库根目录**无 `.env`**（2026-10-27 全库扫描确认无 `.env`/`.key`/`.pem`/`secrets*` 文件）
- `.gitignore` 已预留排除：`.env`、`*.key`、`*.pem`、`secrets*`、`configs/local*.yaml`
- 若未来接入需密钥的 API：密钥放 `.env`（已在 gitignore），文档只登记变量名与申请方式，**不写值**

## 4. 启动命令

```bash
# --- 测试基线 ---
python -m pytest                       # 167 项；约数十秒
python -m pytest -k leakage            # 只跑泄漏闸门

# --- 合成数据冒烟（不接触真实数据） ---
PYTHONPATH=src python -c "
from floodrisk.synthetic import write_synthetic_bundle
print(write_synthetic_bundle('data/synthetic', mode='dynamic', n_grids=64, n_steps=240, seed=42))"

# --- 产品 MVP ---
.venv/bin/python -m hko_api.hko_fetch --loop        # 终端 1：实时抓取（--once 单次）
.venv/bin/python -m http.server 8321 --bind 127.0.0.1  # 终端 2：静态服务（须从项目根目录起）
# http://127.0.0.1:8321/hko_api/dashboard.html      # 整合 Dashboard
# http://127.0.0.1:8321/hko_api/district/           # 18 区播报

# --- 前端规则测试 ---
node hko_api/district/tests/bulletin.test.mjs       # 8 passed / 0 failed
```

## 5. 测试

- `tests/`：15 个测试文件、**167 项**（collect 实测）；最近一次全量通过记录 2026-10-27（迭代#24）
- pytest 配置在 `pyproject.toml`：`testpaths=["tests"]`、`--basetemp=.pytest-tmp`（指向项目内，避免污染系统临时目录）
- 重点覆盖「必须失败的场景真的会失败」：泄漏闸门、切分重叠、无坐标空间切分、标签窗口 purge、固定分级缺 cutoffs 等（README §九）

## 6. 常见故障

| 症状 | 原因与处置 |
|---|---|
| `Library not loaded: @rpath/libomp.dylib` | macOS 缺 OpenMP 运行时 → §2 |
| Dashboard 白屏/数据 404 | http.server 未从项目根目录启动（页面引用相对路径 `data/latest/`、`../data/latest/`） |
| Dashboard 全部显示 STALE | 抓取循环没在跑或 `data/latest/snapshot.json` 过期 >30 min；启动 `hko_fetch --loop` 或接受「打包快照模式」 |
| file:// 双击无数据 | 属预期：file:// 模式用 `snapshot.js`/`grid_scores.js` 打包快照，不自动刷新 |
| pandas 3.x 行为差异报错 | 确认按 `uv.lock` 安装，勿用 pip 自由解析升级 |
| 端口 8321 被占用 | 旧进程未停：`lsof -i :8321` 找到后 `kill` |
| pytest basetemp 权限错误 | `.pytest-tmp/` 在项目内，确认对项目目录可写 |
| 大 parquet 读入内存爆炸 | 用 `realdata.py load_delivery` 的两遍读法（arrow 字典编码 + filter_keys），不要 `pd.read_parquet` 全表裸读 |

## 7. 迁移新机器 checklist

1. 安装 Python 3.12 + libomp + Node
2. 拷贝仓库（**必须包含** `data/delivery_repaired_b1/` 与 `outputs/formal_training_*` 等冻结目录；`hko_api/data/raw/` 280 MB 可不拷，抓取循环会重建）
3. `uv sync`（或 venv+pip）→ `pytest` 验证 167 项
4. 重启 `hko_fetch --loop` 与 `http.server`（先确认旧机器已停）
5. 首次抓取后检查 `hko_api/data/latest/snapshot.json` 五源 `stale:false`
