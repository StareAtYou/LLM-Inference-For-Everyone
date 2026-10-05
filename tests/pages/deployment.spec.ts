import { test, expect } from "@playwright/test";

test("Pages supports navigation, lazy assets and refreshing shared deep links", async ({
  page,
  baseURL,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (
      new URL(response.url()).origin === new URL(baseURL!).origin &&
      response.status() >= 400
    )
      errors.push(`${response.status()} ${response.url()}`);
  });
  await page.goto("./");
  await page.getByRole("link", { name: "开始探索", exact: true }).click();
  await expect(page).toHaveURL(/\/LLM-Inference-For-Everyone\/#\/learn/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("地图");
  await page.goto("./#/learn/rope?depth=expert");
  await expect(page.getByTestId("mechanism-player")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("mechanism-player")).toBeVisible();
  await expect(page).toHaveURL(/#\/learn\/rope\?depth=expert/);
  const icon = await page.locator('link[rel="icon"]').getAttribute("href");
  expect(icon).toBe("/LLM-Inference-For-Everyone/favicon.svg");
  expect((await page.request.get(icon!)).status()).toBe(200);
  expect(errors).toEqual([]);
});

test("Pages inference animation plays after a direct deep-link load", async ({
  page,
}) => {
  await page.goto("./#/pipeline?depth=beginner");
  await page
    .getByRole("button", { name: "连续演示全过程", exact: true })
    .click();
  const board = page.getByLabel("全过程数据流", { exact: true });
  await expect(page.getByTestId("continuous-journey-scene")).toBeVisible();
  await page.getByLabel("播放速度").selectOption("0.25");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await expect
    .poll(async () => Number(await board.getAttribute("data-position")))
    .toBeGreaterThan(0.05);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.reload();
  await expect(page.getByTestId("trace-stage")).toBeVisible();
});
