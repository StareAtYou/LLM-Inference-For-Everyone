import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
const directory = "docs/quality/screenshots/motion";
await mkdir(directory, { recursive: true });
const browser = await chromium.launch({
  channel: process.platform === "darwin" ? "chrome" : "chromium",
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
const base = process.env.PREVIEW_URL || "http://127.0.0.1:5173";
await page.goto(base + "/pipeline?depth=beginner");
await page.getByRole("button", { name: "连续演示全过程", exact: true }).click();
await page
  .getByRole("button", { name: "连续轨迹第 4 层", exact: true })
  .click();
await page.getByLabel("播放速度").selectOption("0.25");
await page.locator(".cj-canvas").scrollIntoViewIfNeeded();
await page.getByRole("button", { name: "播放", exact: true }).click();
for (let i = 0; i < 6; i++) {
  await page.waitForTimeout(450);
  await page
    .locator(".cj-canvas")
    .screenshot({ path: `${directory}/layer-${i}.png` });
}
await page.getByRole("button", { name: "暂停", exact: true }).click();
await page.getByRole("button", { name: "跳到输出与反馈" }).click();
await page.getByLabel("播放速度").selectOption("1");
await page.locator(".cj-canvas").scrollIntoViewIfNeeded();
await page.getByRole("button", { name: "播放", exact: true }).click();
for (let i = 0; i < 3; i++) {
  await page.waitForTimeout(300);
  await page
    .locator(".cj-canvas")
    .screenshot({ path: `${directory}/feedback-${i}.png` });
}
await page.getByRole("button", { name: "暂停", exact: true }).click();
for (const topic of ["rope", "attention", "moe", "kv-cache"]) {
  await page.goto(base + "/learn/" + topic);
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  await page.locator(".cm-canvas").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "播放原理动图", exact: true }).click();
  await page.waitForTimeout(2000);
  await page.getByRole("button", { name: "暂停原理动图", exact: true }).click();
  await page
    .getByTestId("continuous-mechanism-scene")
    .screenshot({ path: `${directory}/${topic}.png` });
}
await page.setViewportSize({ width: 360, height: 900 });
await page.goto(base + "/learn/rope");
await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
await page.getByTestId("continuous-mechanism-scene").scrollIntoViewIfNeeded();
await page
  .getByTestId("continuous-mechanism-scene")
  .screenshot({ path: `${directory}/mobile-rope.png` });
await page.goto(base + "/pipeline?mode=batch");
await page.getByRole("button", { name: "连续演示全过程", exact: true }).click();
await page
  .getByRole("button", { name: "连续轨迹第 1 层", exact: true })
  .click();
await page
  .getByTestId("continuous-journey-scene")
  .screenshot({ path: `${directory}/mobile-batch.png` });
await browser.close();
console.log(
  `Captured temporal layer/feedback samples and mechanism/mobile scenes in ${directory}`,
);
