import { test, expect } from "@playwright/test";
test("home_navigation_keyboard_mobile", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("推理");
  await expect(
    page.getByRole("link", { name: "开始探索", exact: true }),
  ).toHaveAttribute("href", "/learn");
  await page.setViewportSize({ width: 390, height: 844 });
  const menu = page.getByRole("button", { name: "展开导航" });
  await menu.click();
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await page.keyboard.press("Escape");
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await page.goto("/missing");
  await expect(page.getByRole("link", { name: "返回首页" })).toBeVisible();
});
