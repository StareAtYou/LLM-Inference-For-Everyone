import { test, expect } from "@playwright/test";
const routes = [
  "/",
  "/learn",
  "/learn/kv-cache?depth=expert",
  "/pipeline",
  "/models",
  "/models?model=qwen36-moe",
  "/lab/kv-cache",
  "/lab/paged-attention",
  "/lab/continuous-batching",
  "/lab/quantization",
  "/lab/speculative-decoding",
  "/frameworks/vllm?module=runner",
  "/frameworks/sglang?module=cache",
];
for (const width of [360, 390, 768, 1440, 1920])
  test(`direct_routes_are_finite_and_fit_${width}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    for (const route of routes) {
      await page.goto(route);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await expect(page.locator("main")).not.toContainText(/NaN|Infinity/);
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
        route,
      ).toBeLessThanOrEqual(width);
      await page.reload();
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    }
    expect(errors).toEqual([]);
  });
test("experiment_state_updates_and_clock_pauses_offscreen", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/lab/quantization");
  const error = page.locator('[data-testid="quant-error"]');
  await expect(error).toHaveText("0.001853");
  await page.getByRole("button", { name: "INT8", exact: true }).click();
  await expect(error).toHaveText("0.000003");
  await page.getByLabel("权重场景").selectOption("zero");
  await expect(error).toHaveText("0.000000");
  await page.goto("/pipeline");
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.locator(".site-footer").scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  await page.goto("/lab/speculative-decoding");
  await page.getByRole("button", { name: "单步" }).click();
  await expect(page.locator('[data-testid="spec-output"]')).toHaveText(
    "模型先生成",
  );
  await page.getByLabel("草稿场景").selectOption("low");
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    "1 / 13",
  );
});
test("extreme_parameters_storage_and_keyboard_are_recoverable", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem("inference-atlas:preferences", "{"),
  );
  await page.goto(
    "/lab/kv-cache?model=qwen36-moe&tokens=262144&batch=64&bytes=4",
  );
  await expect(page.locator('[data-testid="kv-bytes"]')).toHaveText(
    "640.00 GiB",
  );
  await page.goto("/learn?model=nope&depth=nope");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("地图");
  await page.keyboard.press("Control+k");
  await expect(page.getByRole("searchbox")).toBeFocused();
  await page.getByRole("searchbox").fill("missing term");
  await expect(page.getByText("没有找到相关主题")).toBeVisible();
  await page.getByRole("button", { name: "清除筛选" }).click();
  await expect(
    page.getByRole("link", { name: "KV Cache：复用历史计算", exact: true }),
  ).toBeVisible();
  await page.goto("/frameworks/vllm?module=cache&module=runner");
  await expect(page.getByText("模块参数无效，已回到请求入口。")).toBeVisible();
});

test("background_pause_font_fallback_and_text_zoom", async ({ page }) => {
  await page.route("**/*.woff2", (route) => route.abort());
  await page.goto("/pipeline");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      value: true,
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  const position = await page
    .locator('[data-testid="frame-position"]')
    .textContent();
  await page.waitForTimeout(1050);
  await expect(page.locator('[data-testid="frame-position"]')).toHaveText(
    position!,
  );
  await page.goto("/frameworks/vllm?module=runner");
  await expect(page.locator(".source-inspector h2")).toHaveText(
    "GPUModelRunner",
  );
  await page.evaluate(() => {
    const elements = [...document.querySelectorAll("body *")].filter(
      (e) =>
        e instanceof HTMLElement &&
        [...e.childNodes].some(
          (n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim(),
        ),
    ) as HTMLElement[];
    const sizes = elements.map((e) => parseFloat(getComputedStyle(e).fontSize));
    elements.forEach((e, i) => (e.style.fontSize = sizes[i] * 2 + "px"));
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(1280);
});

test("tablet_layers_stay_inside_their_canvas", async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1000 });
  await page.goto("/models");
  await expect(page.locator(".layer-cell")).toHaveCount(64);
  const bounds = await page.locator(".layer-grid").evaluate((e) => ({
    right: e.getBoundingClientRect().right,
    children: [...e.children].map((c) => c.getBoundingClientRect().right),
  }));
  expect(Math.max(...bounds.children)).toBeLessThanOrEqual(bounds.right + 1);
});

test("mobile_text_zoom_keeps_home_and_frameworks_in_bounds", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 1000 });
  for (const route of ["/", "/frameworks/sglang?module=cache"]) {
    await page.goto(route);
    await page.getByRole("heading", { level: 1 }).waitFor();
    await page.evaluate(() => {
      const es = [...document.querySelectorAll("body *")].filter(
        (e) =>
          e instanceof HTMLElement &&
          [...e.childNodes].some(
            (n) => n.nodeType === Node.TEXT_NODE && n.textContent?.trim(),
          ),
      ) as HTMLElement[];
      const sizes = es.map((e) => parseFloat(getComputedStyle(e).fontSize));
      es.forEach((e, i) => (e.style.fontSize = sizes[i] * 2 + "px"));
    });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      route,
    ).toBeLessThanOrEqual(390);
  }
});

test("paging_extreme_param_matches_control_and_fails_atomically", async ({
  page,
}) => {
  await page.goto("/lab/paged-attention?tokens=262144&capacity=1&blockSize=1");
  await expect(page.getByLabel("新请求 Token 数")).toHaveValue("262144");
  await page.getByRole("button", { name: /分配请求/ }).click();
  await expect(page.getByRole("alert")).toContainText("空闲块不足");
  await expect(page.locator(".physical-block.occupied")).toHaveCount(0);
});
