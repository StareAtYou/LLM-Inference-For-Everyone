import { chromium } from "@playwright/test";
import fs from "node:fs";
const base = process.env.PREVIEW_URL || "http://127.0.0.1:4173";
const out = "docs/quality/workbench-screenshots";
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({
  channel: process.platform === "darwin" ? "chrome" : "chromium",
});
const results = [];
for (const width of [360, 390, 768, 1440, 1920]) {
  const page = await browser.newPage({
    viewport: { width, height: 1000 },
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const state of [
    "single",
    "batch",
    "operator",
    "released",
    "attention",
    "topic",
  ]) {
    const route =
      state === "topic"
        ? "/learn/rope?depth=advanced"
        : state === "attention"
          ? "/pipeline?view=mechanisms"
          : "/pipeline?depth=beginner" +
            (state === "single" ? "" : "&mode=batch");
    await page.goto(base + route);
    await page.getByRole("heading", { level: 1 }).waitFor();
    await page.evaluate(() => document.fonts.ready);
    if (state === "operator") {
      await page.getByLabel("定位网络层").selectOption("3");
      await page.getByRole("button", { name: "单步", exact: true }).click();
    }
    if (state === "released")
      await page.getByRole("button", { name: "跳到结束与释放" }).click();
    if (state === "attention") {
      await page
        .getByRole("searchbox", { name: "搜索原理动图" })
        .fill("Attention");
      await page.getByRole("button", { name: /打开 Attention ·/ }).click();
      await page
        .getByTestId("mechanism-player")
        .locator(".mp-stage-rail button")
        .nth(2)
        .click();
    }
    if (state === "topic")
      await page.getByRole("button", { name: "单步原理动图" }).click();
    await page.evaluate(() => window.scrollTo(0, 0));
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    );
    await page.screenshot({
      path: `${out}/${state}-${width}.png`,
      fullPage: true,
    });
    results.push({ width, state, overflow, errors: [...errors] });
  }
  await page.close();
}
await browser.close();
fs.writeFileSync(`${out}/checks.json`, JSON.stringify(results, null, 2));
console.log(
  JSON.stringify(
    {
      checks: results.length,
      failures: results.filter((r) => r.overflow || r.errors.length),
    },
    null,
    2,
  ),
);
