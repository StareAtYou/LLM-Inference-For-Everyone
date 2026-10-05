import { test, expect } from "@playwright/test";
test("cache_share_release_and_invalid_parameters", async ({ page }) => {
  await page.goto("/lab/kv-cache?tokens=4096&batch=2&bytes=2");
  await expect(page.locator('[data-testid="kv-bytes"]')).toHaveText(
    "512.00 MiB",
  );
  await expect(
    page.getByText("全注意力 KV 分项", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByLabel("上下文 Token 数")).toHaveValue("4096");
  await page.goto("/lab/kv-cache?tokens=Infinity&bytes=3");
  await expect(
    page.getByText("部分实验参数无效，已恢复为默认值。"),
  ).toBeVisible();
  await page.goto("/lab/paged-attention?capacity=3&blockSize=4&tokens=5");
  await page.getByRole("button", { name: "分配请求 · 5 tokens" }).click();
  await expect(page.getByRole("button", { name: "释放 R1" })).toBeVisible();
  await page.getByRole("button", { name: "分配请求 · 5 tokens" }).click();
  await expect(page.getByRole("alert")).toContainText("空闲块不足");
  await page.getByRole("button", { name: "释放 R1" }).click();
  await page.getByRole("button", { name: "分配请求 · 5 tokens" }).click();
  await expect(page.getByRole("button", { name: "释放 R2" })).toBeVisible();
});
