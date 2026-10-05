import { chromium } from "@playwright/test";
import fs from "node:fs";
const base = process.env.PREVIEW_URL || "http://127.0.0.1:5173";
const out = process.env.SCREENSHOT_DIR || "docs/quality/screenshots";
fs.mkdirSync(out, { recursive: true });
const routes = [
  ["home", "/"],
  ["learn", "/learn"],
  ["topic", "/learn/kv-cache?depth=expert"],
  ["pipeline", "/pipeline"],
  ["dense", "/models?layer=3"],
  ["moe", "/models?model=qwen36-moe"],
  ["linear", "/models?depth=advanced"],
  ["teaching", "/models?model=teaching"],
  ["kv", "/lab/kv-cache"],
  ["paging", "/lab/paged-attention"],
  ["batching", "/lab/continuous-batching"],
  ["quantization", "/lab/quantization"],
  ["speculation", "/lab/speculative-decoding"],
  ["vllm", "/frameworks/vllm?module=scheduler"],
  ["sglang", "/frameworks/sglang?module=cache"],
];
const browser = await chromium.launch({
  channel:
    process.env.PLAYWRIGHT_CHANNEL ||
    (process.platform === "darwin" ? "chrome" : "chromium"),
});
const results = [];
for (const width of [390, 768, 1440, 360, 1920]) {
  const page = await browser.newPage({
    viewport: { width, height: 1000 },
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const [name, route] of routes) {
    await page.goto(base + route);
    await page.getByRole("heading", { level: 1 }).waitFor();
    await page.evaluate(() => document.fonts.ready);
    if (name === "linear") {
      await page
        .getByRole("button", { name: "更新一个 Token", exact: true })
        .click();
      await page
        .getByRole("button", { name: "更新一个 Token", exact: true })
        .click();
    }
    if (name === "teaching") {
      await page.getByLabel("教学 Q 头数").selectOption("8");
      await page.getByLabel("教学 KV 头数").selectOption("1");
      await page.getByLabel("教学序列长度").fill("12");
    }
    if (name === "paging") {
      await page.getByRole("button", { name: "分配请求 · 5 tokens" }).click();
      await page.getByRole("button", { name: "释放 R1" }).waitFor();
    }
    if (name === "pipeline")
      await page.locator(".stage-rail button").nth(3).click();
    if (["pipeline", "batching", "speculation"].includes(name)) {
      await page.getByRole("button", { name: "单步" }).click();
      await page.getByRole("button", { name: "单步" }).click();
    }
    const dimension = await page.evaluate(() => ({
      viewport: innerWidth,
      document: document.documentElement.scrollWidth,
    }));
    results.push({ name, width, ...dimension, errors: [...errors] });
    if ([390, 768, 1440].includes(width))
      await page.screenshot({
        path: `${out}/${name}-${width}.jpg`,
        type: "jpeg",
        quality: 80,
        fullPage: true,
      });
    if (dimension.document > width)
      console.log("OVERFLOW", name, width, dimension.document);
  }
  await page.close();
  console.log("Checked width", width);
}
fs.writeFileSync(`${out}/checks.json`, JSON.stringify(results, null, 2));
await browser.close();
console.log("Captures complete:", results.length);
