import { test, expect } from "@playwright/test";
test("model_switch_updates_structure_and_experts", async ({ page }) => {
  await page.goto("/models");
  await expect(page.locator(".layer-cell")).toHaveCount(64);
  await page.getByRole("button", { name: "第 4 层 · Full Attention" }).click();
  await expect(page.locator(".block-inspector")).toContainText(
    "Full Attention",
  );
  await expect(page.getByLabel("隐藏维度")).toBeDisabled();
  await page.getByLabel("选择模型").selectOption("qwen36-moe");
  await expect(page.locator(".layer-cell")).toHaveCount(40);
  await expect(page.locator(".expert-summary")).toContainText(
    "8 routed + 1 shared",
  );
  await page.getByLabel("选择模型").selectOption("teaching");
  await page.getByLabel("隐藏维度").fill("192");
  await expect(page.getByText("自定义教学配置", { exact: true })).toBeVisible();
  await page.getByLabel("选择模型").selectOption("qwen38-dense");
  await expect(page.getByLabel("隐藏维度")).toBeDisabled();
});
