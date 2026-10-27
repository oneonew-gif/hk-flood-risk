// T6 规则模板单元测试：node hko_api/district/tests/bulletin.test.mjs
import fs from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const code = fs.readFileSync(path.join(here, "..", "bulletin.js"), "utf8");
const sandbox = { window: {}, console };
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const { buildBulletin, rainBand } = sandbox.window;

let pass = 0;
const fails = [];
function check(name, fn) {
  try {
    fn();
    pass += 1;
    console.log("PASS", name);
  } catch (e) {
    fails.push(`${name}: ${e.message}`);
    console.log("FAIL", name, "-", e.message);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// 1. 雨量分级边界
check("rainBand 边界", () => {
  const cases = [[0, "無錄得降雨"], [0.4, "微量降雨"], [1, "小雨"], [4.9, "小雨"],
    [5, "中等降雨"], [14.9, "中等降雨"], [15, "大雨"], [29.9, "大雨"], [30, "暴雨"], [120, "暴雨"]];
  for (const [mm, label] of cases) {
    assert(rainBand(mm).label === label, `${mm} 应为 ${label}，实际 ${rainBand(mm).label}`);
  }
  assert(rainBand(null) === null, "null 应返回 null");
});

// 2. 正常播报：0 毫米
check("正常播报（0 mm）", () => {
  const b = buildBulletin({
    place: "沙田", mm: 0, window: "11:45 – 12:45", obsTime: "2026-10-27T13:02:00+08:00",
    fetchTime: "2026-10-27T13:53:51+08:00", warnings: {},
  });
  assert(b.level === "官方觀測", `level=${b.level}`);
  assert(b.headline.includes("沙田") && b.headline.includes("無錄得降雨"), b.headline);
  const text = b.body.join(" ");
  assert(text.includes("0 毫米"), "应包含官方数值 0 毫米");
  assert(text.includes("11:45 – 12:45"), "应包含统计窗口");
  assert(text.includes("全港現時沒有生效"), "应说明全港无警告");
  assert(text.includes("不含未來降雨預測"), "应保留免责声明");
});

// 3. 中等降雨 + 红色暴雨警告
check("中等降雨 + 全港红色暴雨警告", () => {
  const b = buildBulletin({
    place: "元朗", mm: 12.3, warnings: { WRAINR: { code: "WRAINR", name: "Red Rainstorm Warning" } },
  });
  const text = b.body.join(" ");
  assert(b.headline.includes("中等降雨"), b.headline);
  assert(text.includes("紅色暴雨警告"), "应翻译警告名称");
  assert(text.includes("全港級資訊"), "警告必须标注为全港级");
});

// 4. 缺测：不得用邻区或平均值替代
check("缺测不推算", () => {
  const b = buildBulletin({ place: "南區", mm: null, warnings: {} });
  assert(b.level === "缺測", b.level);
  const text = b.headline + " " + b.body.join(" ");
  assert(text.includes("缺測"), "应说明缺测");
  assert(!text.includes("毫米"), "不得出现任何毫米数值");
  assert(text.includes("不以鄰區數值"), "应声明不替代");
});

// 5. 接口失败：不编数值
check("接口失败不编数值", () => {
  const b = buildBulletin({ place: "東區", ok: false, mm: 8 });
  assert(b.tone === "err", b.tone);
  const text = b.body.join(" ");
  assert(text.includes("未返回有效數據"), "应说明接口失败");
  assert(!text.includes("8 毫米"), "失败时不得显示旧/猜测数值");
});

// 6. 过期快照：强制前缀标注
check("过期快照标注", () => {
  const b = buildBulletin({
    place: "葵青", mm: 2.5, stale: true, staleReason: "快照獲取於 45 分鐘前且未再成功更新",
    warnings: {},
  });
  assert(b.level === "過期快照", b.level);
  assert(b.body[0].startsWith("⚠"), "第一条必须是过期警示");
  assert(b.body[0].includes("不代表現時天氣"), "必须声明不代表现时天气");
  assert(b.body.join(" ").includes("2.5 毫米"), "过期仍可显示最后快照数值");
});

// 7. 无任何警告 vs 多警告
check("多警告并列", () => {
  const b = buildBulletin({
    place: "灣仔", mm: 1,
    warnings: { WRAINA: { code: "WRAINA" }, WTS: { code: "WTS" } },
  });
  const text = b.body.join(" ");
  assert(text.includes("黃色暴雨警告") && text.includes("雷暴警告"), text);
});

// 8. 输出不含预报数值
check("不含预报数值", () => {
  const b = buildBulletin({ place: "觀塘", mm: 3, warnings: {} });
  const text = (b.headline + " " + b.body.join(" ")).toLowerCase();
  for (const bad of ["mm/30min", "未來 2 小時降雨", "水浸概率", "水浸風險數值"]) {
    assert(!text.includes(bad.toLowerCase()), `不应包含「${bad}」`);
  }
});

console.log(`\n${pass} passed, ${fails.length} failed`);
if (fails.length) {
  console.error(fails.join("\n"));
  process.exit(1);
}
