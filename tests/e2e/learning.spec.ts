import { test, expect } from "@playwright/test";
test("learning_navigation_and_recovery", async ({ page }) => {
  await page.goto("/learn?depth=beginner");
  await page
    .getByRole("searchbox", { name: "搜索知识与术语" })
    .fill("KV Cache");
  await page
    .getByRole("link", { name: "KV Cache：复用历史计算", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "KV Cache",
  );
  const content = await page.locator(".topic-explanation").textContent();
  await page.getByRole("button", { name: "精通", exact: true }).click();
  await expect(page.locator(".topic-explanation")).not.toHaveText(content!);
  await page.getByRole("button", { name: "收藏主题" }).click();
  await page.reload();
  await expect(page.getByRole("button", { name: "取消收藏" })).toBeVisible();
});
test("disabled_storage_keeps_learning_usable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new Error("disabled");
      },
    });
  });
  await page.goto("/learn");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("地图");
  await page
    .getByRole("searchbox", { name: "搜索知识与术语" })
    .fill("no-such-topic");
  await expect(page.getByText("没有找到相关主题")).toBeVisible();
});
