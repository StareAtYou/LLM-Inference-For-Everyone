import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const directory = "docs/quality/screenshots/distributed";
await mkdir(directory, { recursive: true });
const base = process.env.PREVIEW_URL || "http://127.0.0.1:5173";
const browser = await chromium.launch({
  channel: process.platform === "darwin" ? "chrome" : "chromium",
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
for (const width of [390, 1440]) {
  await page.setViewportSize({ width, height: 1080 });
  for (const [name, path, target] of [
    [
      "parallel",
      "/distributed?parallel=tp&depth=advanced",
      "parallel-explorer",
    ],
    [
      "collectives",
      "/distributed?view=collectives&collective=all-to-all&depth=advanced",
      "collective-explorer",
    ],
    ["shapes", "/learn/attention?depth=expert", "shape-teaching"],
  ]) {
    await page.goto(base + path);
    await page.getByTestId(target).first().waitFor({ state: "visible" });
    await page.evaluate(() => document.fonts.ready);
    const documentWidth = await page.evaluate(
      () => document.documentElement.scrollWidth,
    );
    if (documentWidth > width) throw new Error(`${name} overflows ${width}px`);
    await page.screenshot({
      path: `${directory}/${name}-${width}.png`,
      fullPage: true,
    });
  }
}
await browser.close();
if (errors.length) throw new Error(errors.join("\n"));
console.log(`Captured six shape/distributed layouts in ${directory}`);
