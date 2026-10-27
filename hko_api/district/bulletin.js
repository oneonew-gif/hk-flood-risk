"use strict";
/**
 * T6 · 香港 18 区气象播报规则模板
 * 纯函数：不访问网络、不读时钟、不生成任何预报数值。
 * 输入全部来自 HKO 官方快照（rhrread.districts / warnings）。
 */

const DISTRICT_RAIN_BANDS = [
  { max: 0.0, label: "無錄得降雨", desc: "暫無錄得降雨" },
  { max: 1.0, label: "微量降雨", desc: "僅錄得微量降雨" },
  { max: 5.0, label: "小雨", desc: "有零星小雨" },
  { max: 15.0, label: "中等降雨", desc: "有中等程度降雨" },
  { max: 30.0, label: "大雨", desc: "有大雨" },
  { max: Infinity, label: "暴雨", desc: "有暴雨" },
];

/** 分段口径：0 = 无；0–1 微量；1–5 小；5–15 中；15–30 大；≥30 暴雨 */
function rainBand(mm) {
  if (mm === null || mm === undefined || Number.isNaN(mm)) return null;
  if (mm <= 0) return DISTRICT_RAIN_BANDS[0];
  if (mm < 1) return DISTRICT_RAIN_BANDS[1];
  if (mm < 5) return DISTRICT_RAIN_BANDS[2];
  if (mm < 15) return DISTRICT_RAIN_BANDS[3];
  if (mm < 30) return DISTRICT_RAIN_BANDS[4];
  return DISTRICT_RAIN_BANDS[5];
}

/** 取香港天文台官方全港警告列表（只做翻译，不做推断） */
const WARNING_LABELS = {
  WRAINA: "黃色暴雨警告",
  WRAINR: "紅色暴雨警告",
  WRAINB: "黑色暴雨警告",
  WFNTSA: "新界北部水浸特別提示",
  WTS: "雷暴警告",
  WL: "山泥傾瀉警告",
  WTCSGNL: "熱帶氣旋警告",
  WMSGNL: "強烈季候風信號",
};

function warningNames(warnings) {
  if (!warnings) return [];
  return Object.values(warnings).map((w) => WARNING_LABELS[w.code] || w.name || w.code);
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString("zh-HK", { timeZone: "Asia/Hong_Kong", hour12: false });
}

/**
 * 生成地区气象播报。
 * @param {object} ctx
 * @param {string} ctx.place        区名（HKO 官方名称）
 * @param {number|null} ctx.mm      过去 1 小时该区最大雨量（mm）；不可得时传 null
 * @param {string|null} ctx.window   官方统计窗口文字
 * @param {string|null} ctx.obsTime  官方观测时刻
 * @param {string|null} ctx.fetchTime 数据获取时刻
 * @param {boolean} ctx.stale       是否显示的是过期/失败快照
 * @param {string} ctx.staleReason  过期/失败原因
 * @param {object|null} ctx.warnings 全港警告对象
 * @param {boolean} ctx.ok          源是否可用
 * @returns {{headline:string, body:string[], level:string, tone:string}}
 */
function buildBulletin(ctx) {
  const {
    place, mm = null, window = null, obsTime = null, fetchTime = null,
    stale = false, staleReason = "", warnings = null, ok = true,
  } = ctx || {};

  const name = place || "所選地區";

  // 1) 数据不可用：明确说明，不编数值
  if (!ok) {
    return {
      level: "無數據",
      tone: "err",
      headline: `${name}：氣象數據暫不可用`,
      body: [
        "香港天文台自動氣象站／分區雨量接口本次未返回有效數據。",
        "基於規則：不顯示任何推算或替代數值，請稍後重試或查看官方網站。",
      ],
    };
  }

  // 2) 过期快照：保留数值但强制标注
  const stalePrefix = stale
    ? [`⚠ 以下為最後一次成功獲取的快照（${staleReason || "已過期"}），不代表現時天氣。`]
    : [];

  // 3) 雨量不可得（该区缺失）
  if (mm === null || mm === undefined || Number.isNaN(mm)) {
    return {
      level: "缺測",
      tone: "warn",
      headline: `${name}：過去 1 小時雨量缺測`,
      body: [
        ...stalePrefix,
        "本次官方分區雨量數據中沒有該區的有效數值。",
        "基於規則：不以鄰區數值或全港平均代替，請以香港天文台官方公布為準。",
      ],
    };
  }

  const band = rainBand(mm);
  const winText = window ? `（統計窗口 ${window}）` : "";
  const body = [
    ...stalePrefix,
    `過去 1 小時本區錄得最高雨量 ${mm} 毫米${winText}，屬${band.desc}。`,
  ];

  const wn = warningNames(warnings);
  if (wn.length) {
    body.push(`全港現時生效警告：${wn.join("、")}（全港級資訊，非本區專屬）。`);
  } else {
    body.push("全港現時沒有生效暴雨或相關警告（全港級資訊，非本區專屬）。");
  }

  body.push(`資料來源：香港天文台公開數據；官方觀測時間 ${fmtTime(obsTime)}，數據獲取時間 ${fmtTime(fetchTime)}。`);
  body.push("本播報僅為官方觀測的規則化轉述，不含未來降雨預測，也不代表水浸風險。");

  return {
    level: stale ? "過期快照" : "官方觀測",
    tone: stale ? "warn" : "ok",
    headline: `${name}：${band.label}（${mm} mm / 1 小時）`,
    body,
  };
}

if (typeof window !== "undefined") {
  window.rainBand = rainBand;
  window.warningNames = warningNames;
  window.buildBulletin = buildBulletin;
  window.fmtTime = fmtTime;
}
