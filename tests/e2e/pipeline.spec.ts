import { test, expect } from "@playwright/test";
test("pipeline_clock_is_single_and_resettable", async ({ page }) => {
  await page.goto("/pipeline");
  const step = page.getByRole("button", { name: "单步" });
  const initial = (await page.getByTestId("frame-position").textContent())!;
  const count = initial.split(" / ")[1];
  await step.click();
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    `2 / ${count}`,
  );
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.waitForTimeout(1050);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const value = await page
    .locator('[data-testid="frame-position"]')
    .textContent();
  await page.waitForTimeout(1100);
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    value!,
  );
  await page.getByRole("button", { name: "重置" }).click();
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    `1 / ${count}`,
  );
  await step.click();
  await page.getByLabel("选择模型").selectOption("qwen36-moe");
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    /^1 \/ \d+$/,
  );
  await page.getByRole("link", { name: "学习地图", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("地图");
});
