#!/usr/bin/env python3
"""HKO Open Data 实时数据获取模块（独立模块，纯标准库，不依赖 src/floodrisk）。

数据源（2026-10-26 实测核验，端点清单见 hko_api/reports/hko_api_inventory_and_feature_mapping.md）：
  1. warnings       天气警告摘要（暴雨黄/红/黑、WFNTSA、雷暴、台风信号等）
  2. rhrread        各区过去 1 小时最大雨量 + 实况
  3. hourlyRainfall 自动气象站过去 1 小时雨量
  4. hhot           鲗鱼涌(QUB)逐时天文潮（确定性预报，整年可预取）
  5. nowcast        网格降雨临近预报（未来 2h、半小时步长，HK bbox 裁剪）

存储约定：
  - 原始响应原样保存  data/raw/{source}/{YYYYMMDD}/{source}_{HHMMSS}.{json|csv} + 同名 .meta.json
  - 全部请求追加日志  data/raw/fetch_log.jsonl（获取时间/观测时间/HTTP 状态/异常）
  - 归一化快照        data/latest/snapshot.json + snapshot.js（file:// 双击备用）
  - 网络失败：保留该源上一次成功快照，stale 置 true 并记录 last_error，不崩溃

边界约束：本模块输出仅为 HKO 官方实况/官方预报展示数据，
不构造任何模型输入特征，不调用任何模型 bundle，不生成水浸概率。
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HKO_BASE = "https://data.weather.gov.hk/weatherAPI/opendata"
NOWCAST_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast.csv"
HHOT_STATION = "QUB"  # 鲗鱼涌（维港主站；可扩展 14 站）
HHOT_YEAR = 2026

# 香港粗 bbox（nowcast 网格域远大于香港，须裁剪；实测网格步长 ~0.018°x0.02°）
HK_BBOX = {"lat_min": 22.15, "lat_max": 22.58, "lon_min": 113.83, "lon_max": 114.45}

HKT = timezone(timedelta(hours=8))

# 每源 STALE 阈值（分钟，按观测时间 obs_time 起算；仅标记不删除）
STALE_AFTER_MIN = {
    "warnings": 15,
    "rhrread": 30,
    "hourlyRainfall": 90,
    "hhot": 26 * 60,
    "nowcast": 30,
}

# 轮询间隔（秒）
POLL_INTERVAL_SEC = {
    "warnings": 60,
    "rhrread": 300,
    "hourlyRainfall": 300,
    "hhot": 3600,
    "nowcast": 720,
}

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
LATEST_DIR = ROOT / "data" / "latest"

DISCLAIMER = (
    "本页数据来自香港天文台公开API：警告/雨量/潮位为官方观测，"
    "临近预报为官方产品；地图中的网格评分为历史模型输出"
    "（弱监督标签下探索性风险排序，非实时水浸发生概率）。"
)


def now_hkt() -> datetime:
    return datetime.now(HKT)


def hkt_str(dt: datetime | None = None) -> str:
    return (dt or now_hkt()).strftime("%Y-%m-%dT%H:%M:%S+08:00")


# ---------------------------------------------------------------------------
# HTTP 层：超时 + 重试 + 原始落盘 + 日志
# ---------------------------------------------------------------------------

def _http_get(url: str, source: str, retries: int = 2, timeout: int = 20) -> tuple[bytes, int]:
    """GET 请求，成功返回 (body, status)。失败抛 LastFetchError。"""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hk-flood-risk-mvp/1.0 (educational)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                _save_raw(source, url, body, resp.status, None)
                return body, resp.status
        except Exception as e:  # 网络异常/超时/非 200
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (2**attempt))
    _save_raw(source, url, b"", 0, repr(last_err))
    raise LastFetchError(f"{source}: {last_err!r}")


class LastFetchError(RuntimeError):
    pass


def _save_raw(source: str, url: str, body: bytes, status: int, error: str | None) -> None:
    """原始响应原样落盘 + fetch_log 追加（无论成败，保证可追溯）。"""
    ts = now_hkt()
    day_dir = RAW_DIR / source / ts.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    ext = "json" if ".php" in url and "rformat=csv" not in url else "csv"
    stem = f"{source}_{ts.strftime('%H%M%S')}"
    (day_dir / f"{stem}.{ext}").write_bytes(body)
    meta = {"url": url, "http_status": status, "fetch_time_hkt": hkt_str(ts),
            "bytes": len(body), "error": error}
    (day_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with (RAW_DIR / "fetch_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"source": source, **meta}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 各源解析（原始 JSON/CSV → 归一化 dict）
# ---------------------------------------------------------------------------

def fetch_warnings() -> dict:
    body, status = _http_get(f"{HKO_BASE}/weather.php?dataType=warnsum&lang=tc", "warnings")
    data = json.loads(body.decode("utf-8"))
    obs = None
    for w in (data or {}).values():
        t = w.get("updateTime") or w.get("issueTime")
        if t:
            obs = max(obs or "", t)
    return {"ok": True, "http_status": status, "obs_time": obs or hkt_str(),
            "active": bool(data), "warnings": data or {}}


def fetch_rhrread() -> dict:
    body, status = _http_get(f"{HKO_BASE}/weather.php?dataType=rhrread&lang=tc", "rhrread")
    d = json.loads(body.decode("utf-8"))
    rf = d.get("rainfall", {})
    return {"ok": True, "http_status": status, "obs_time": d.get("updateTime"),
            "rainfall_start": rf.get("startTime"), "rainfall_end": rf.get("endTime"),
            "districts": [{"place": x.get("place"), "max_mm": x.get("max"), "unit": x.get("unit")}
                          for x in rf.get("data", [])],
            "icon": d.get("icon"), "icon_update": d.get("iconUpdateTime"),
            "temperature": d.get("temperature", {}).get("data", [])[:6],
            "humidity": d.get("humidity", {}).get("data", [])[:3]}


def fetch_hourly_rainfall() -> dict:
    body, status = _http_get(f"{HKO_BASE}/hourlyRainfall.php?lang=tc", "hourlyRainfall")
    d = json.loads(body.decode("utf-8"))
    rows = []
    for x in d.get("hourlyRainfall", []):
        try:
            v = float(x.get("value"))
        except (TypeError, ValueError):
            v = None
        rows.append({"station": x.get("automaticWeatherStation"),
                     "id": x.get("automaticWeatherStationID"), "mm": v})
    rows.sort(key=lambda r: (-(r["mm"] if r["mm"] is not None else -1)))
    return {"ok": True, "http_status": status, "obs_time": d.get("obsTime"),
            "unit": "mm", "stations": rows}


def fetch_hhot() -> dict:
    """逐时天文潮（小时末标注：列 01 = 当日 01:00 潮位）。返回过去12h+未来12h序列。"""
    url = f"{HKO_BASE}/opendata.php?dataType=HHOT&station={HHOT_STATION}&year={HHOT_YEAR}&rformat=csv"
    body, status = _http_get(url, "hhot")
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    if header[:2] != ["MM", "DD"]:
        raise LastFetchError(f"hhot: 表头异常 {header[:3]}")
    series = []  # (datetime HKT, meters)
    for row in reader:
        if len(row) < 3 or not row[0].strip().isdigit():
            continue
        month, day = int(row[0]), int(row[1])
        for col_idx in range(2, len(row)):
            v = row[col_idx].strip()
            if not v:
                continue
            hour = col_idx - 1  # 列 01 → 01:00（小时末标注）
            if not 1 <= hour <= 24:
                continue
            try:
                meters = float(v)
            except ValueError:
                continue
            hh = hour % 24
            d = datetime(HHOT_YEAR, month, day, hh, tzinfo=HKT)
            if hour == 24:  # 列 24 = 次日 00:00（小时末标注为 24 时）
                d += timedelta(days=1)
            series.append((d, meters))
    series.sort(key=lambda x: x[0])
    t0 = now_hkt().replace(minute=0, second=0, microsecond=0)
    win = [(d.isoformat(), m) for d, m in series
           if t0 - timedelta(hours=12) <= d <= t0 + timedelta(hours=13)]
    # HHOT 为全年确定性天文预报，obs_time 取获取时刻（数据本身无"观测时刻"概念）
    return {"ok": True, "http_status": status, "obs_time": hkt_str(), "station": HHOT_STATION,
            "unit": "m", "note": "天文潮调和预报（确定性），非实测；列 01–24 为小时末标注",
            "series": win}


def fetch_nowcast() -> dict:
    """网格临近预报：原始 CSV 全量落盘；快照仅保留 HK bbox + 逐步统计。"""
    body, status = _http_get(NOWCAST_URL, "nowcast", retries=1, timeout=30)
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    steps: dict[str, list[float]] = {}
    cells_all_steps = []
    updated = None
    n_hk = 0
    for row in reader:
        if len(row) < 5 or not row[0].isdigit():
            continue
        updated = updated or row[0]
        ending, lat_s, lon_s, mm_s = row[1], row[2], row[3], row[4]
        try:
            lat, lon, mm = float(lat_s), float(lon_s), float(mm_s)
        except ValueError:
            continue
        if not (HK_BBOX["lat_min"] <= lat <= HK_BBOX["lat_max"]
                and HK_BBOX["lon_min"] <= lon <= HK_BBOX["lon_max"]):
            continue
        n_hk += 1
        steps.setdefault(ending, []).append(mm)
        cells_all_steps.append({"e": ending, "la": round(lat, 4), "lo": round(lon, 4), "m": mm})
    if not steps:
        raise LastFetchError("nowcast: HK bbox 内无有效行")
    obs = f"{updated[:4]}-{updated[4:6]}-{updated[6:8]}T{updated[8:10]}:{updated[10:12]}:00+08:00"
    step_stats = []
    for ending in sorted(steps):
        vals = steps[ending]
        e = f"{ending[:4]}-{ending[4:6]}-{ending[6:8]}T{ending[8:10]}:{ending[10:12]}:00+08:00"
        wet = [v for v in vals if v > 0]
        step_stats.append({"ending": e, "n_cells": len(vals),
                           "mean_mm": round(sum(vals) / len(vals), 3),
                           "max_mm": max(vals),
                           "n_wet": len(wet),
                           "wet_frac": round(len(wet) / len(vals), 3)})
    return {"ok": True, "http_status": status, "obs_time": obs,
            "unit": "mm/30min", "note": "HKO 官方临近预报（临时数据），HK bbox 裁剪后 " + str(n_hk) + " 行",
            "steps": step_stats, "cells": cells_all_steps}


FETCHERS = {
    "warnings": fetch_warnings,
    "rhrread": fetch_rhrread,
    "hourlyRainfall": fetch_hourly_rainfall,
    "hhot": fetch_hhot,
    "nowcast": fetch_nowcast,
}


# ---------------------------------------------------------------------------
# 快照组装：成功 → 新数据；失败 → 沿用上次成功数据 + stale 标记
# ---------------------------------------------------------------------------

def load_previous_snapshot() -> dict:
    p = LATEST_DIR / "snapshot.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def parse_obs_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def build_snapshot(only: list[str] | None = None) -> dict:
    prev = load_previous_snapshot()
    prev_sources = prev.get("sources", {})
    snap = {"generated_at": hkt_str(), "timezone": "Asia/Hong_Kong",
            "disclaimer": DISCLAIMER, "sources": {}}
    for name, fn in FETCHERS.items():
        if only and name not in only:
            continue
        entry = prev_sources.get(name, {})
        try:
            data = fn()
            entry.update(data)
            entry["stale"] = False
            entry["last_error"] = None
        except LastFetchError as e:
            # 保留上一次成功快照，明确标记过期
            entry["stale"] = True
            entry["last_error"] = str(e)
            entry["error_at"] = hkt_str()
            if not entry:
                entry = {"ok": False, "stale": True, "data": None,
                         "last_error": str(e), "error_at": hkt_str()}
        entry.setdefault("ok", False)
        entry["max_age_min"] = STALE_AFTER_MIN[name]
        # fetch_time 语义 = 本源快照最后一次成功获取的时刻（管道健康度）。
        # 成功 → 更新为本次获取时刻；失败 → 保留上次成功时刻，供客户端计算"已 X 分钟未更新"。
        if entry.get("stale"):
            entry["fetch_time"] = entry.get("fetch_time") or hkt_str()
        else:
            entry["fetch_time"] = hkt_str()
        # 服务端按观测时间重算 stale（快照若长期未更新，写入时就该被标记）
        obs_dt = parse_obs_time(entry.get("obs_time"))
        if obs_dt is not None and not entry.get("ok"):
            entry["stale"] = True
        snap["sources"][name] = entry
    return snap


def write_snapshot(snap: dict) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    (LATEST_DIR / "snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    # file:// 双击演示备用（fetch 受 CORS 限制时走此文件）
    (LATEST_DIR / "snapshot.js").write_text(
        "window.HKO_SNAPSHOT = " + json.dumps(snap, ensure_ascii=False) + ";",
        encoding="utf-8")


def is_stale(entry: dict) -> bool:
    """按观测时间判断过期（供循环内主动复查）。"""
    obs_dt = parse_obs_time(entry.get("obs_time"))
    if obs_dt is None:
        return True
    age_min = (now_hkt() - obs_dt).total_seconds() / 60
    return age_min > entry.get("max_age_min", STALE_AFTER_MIN.get("nowcast", 30))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_once(only: list[str] | None = None) -> dict:
    prev = load_previous_snapshot()
    snap = build_snapshot(only)
    if only and prev.get("sources"):
        # 增量抓取：仅替换本次抓取的源，其余源沿用上次快照（深合并）
        snap["sources"] = {**prev.get("sources", {}), **snap["sources"]}
        snap["timezone"] = prev.get("timezone", snap["timezone"])
        snap["disclaimer"] = prev.get("disclaimer", snap["disclaimer"])
    write_snapshot(snap)
    summary = {k: ("OK" if not v.get("stale") else "STALE")
               for k, v in snap.get("sources", {}).items()}
    print(f"[{hkt_str()}] snapshot written: {summary}", flush=True)
    return snap


def run_loop() -> None:
    print("HKO fetch loop started (Ctrl+C to stop). Poll intervals (s): "
          + json.dumps(POLL_INTERVAL_SEC), flush=True)
    last_run = {k: 0.0 for k in FETCHERS}
    while True:
        now_ts = time.time()
        due = [k for k, iv in POLL_INTERVAL_SEC.items() if now_ts - last_run[k] >= iv]
        if due:
            run_once(due)
            for k in due:
                last_run[k] = time.time()
        time.sleep(5)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--once"
    if mode == "--loop":
        run_loop()
    else:
        only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        run_once(only)
