import { chromium } from "@playwright/test";
import fs from "node:fs";
const base = process.env.PREVIEW_URL || "http://127.0.0.1:4173";
const out = "docs/quality/continuous-screenshots";
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({
  channel: process.platform === "darwin" ? "chrome" : "chromium",
});
const checks = [];
for (const width of [360, 390, 768, 1440, 1920]) {
  const page = await browser.newPage({ viewport: { width, height: 1000 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const state of ["single", "batch", "moe"]) {
    await page.goto(
      base +
        (state === "moe"
          ? "/pipeline?view=mechanisms"
          : "/pipeline?depth=beginner" +
            (state === "batch" ? "&mode=batch" : "")),
    );
    await page.getByRole("heading", { level: 1 }).waitFor();
    if (state === "moe") {
      await page.getByRole("button", { name: /打开 MoE ·/ }).click();
      await page
        .getByRole("button", { name: "连续演示原理", exact: true })
        .click();
      await page
        .getByTestId("mechanism-player")
        .locator(".mp-stage-rail button")
        .nth(1)
        .click();
      await page.getByLabel("连续原理动画进度").press("ArrowRight");
    } else {
      await page
        .getByRole("button", { name: "连续演示全过程", exact: true })
        .click();
      await page.getByLabel("定位网络层").selectOption("3");
      for (let i = 0; i < 30; i++)
        await page.getByLabel("连续演示进度").press("ArrowRight");
    }
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({
      path: `${out}/${state}-${width}.png`,
      fullPage: true,
    });
    checks.push({
      width,
      state,
      overflow: await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
      errors: [...errors],
    });
  }
  await page.close();
}
await browser.close();
fs.writeFileSync(`${out}/checks.json`, JSON.stringify(checks, null, 2));
console.log(
  JSON.stringify(
    {
      checks: checks.length,
      failures: checks.filter((c) => c.overflow || c.errors.length),
    },
    null,
    2,
  ),
);
