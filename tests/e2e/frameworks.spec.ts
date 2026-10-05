import { test, expect } from "@playwright/test";
test("framework_modules_are_versioned_and_shareable", async ({ page }) => {
  await page.goto("/frameworks/sglang?module=cache");
  await expect(page.locator(".source-inspector h2")).toHaveText("RadixCache");
  await page.reload();
  await expect(page.locator(".source-inspector h2")).toHaveText("RadixCache");
  await expect(
    page.getByRole("link", { name: "在 GitHub 查看源码" }),
  ).toHaveAttribute("href", /e00930c5489053f26d86b179cee0d087f846acbb/);
  await page
    .locator(".module-topics")
    .getByRole("link", { name: "Prefix Caching" })
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("前缀");
  await page.goto("/frameworks/vllm?module=blocks");
  await expect(page.locator(".module-responsibility")).toContainText("块哈希");
  await page
    .locator(".framework-tabs")
    .getByRole("link", { name: "SGLang v0.5.21" })
    .click();
  await expect(page.locator(".source-inspector h2")).toHaveText("HTTP Server");
  await page.goto("/frameworks/vllm?module=not-a-module");
  await expect(page.getByText("模块参数无效，已回到请求入口。")).toBeVisible();
});
