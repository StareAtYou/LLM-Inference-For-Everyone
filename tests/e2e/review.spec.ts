import { test, expect } from "@playwright/test";
test("internal_concept_navigation_preserves_disabled_storage_session", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(window, "localStorage", {
      get() {
        throw Error("disabled");
      },
    }),
  );
  await page.goto("/learn/kv-cache");
  await page.getByRole("button", { name: "收藏主题", exact: true }).click();
  await page.getByRole("button", { name: "精通", exact: true }).click();
  await page
    .locator(".main-nav")
    .getByRole("link", { name: "模型结构" })
    .click();
  await page.getByRole("link", { name: "仅解释概念" }).click();
  await expect(
    page.getByRole("button", { name: "精通", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page
    .locator(".main-nav")
    .getByRole("link", { name: "学习地图" })
    .click();
  await page.getByRole("searchbox").fill("KV Cache");
  await page
    .getByRole("link", { name: "KV Cache：复用历史计算", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "取消收藏", exact: true }),
  ).toBeVisible();
});
test("depth_guidance_adds_lab_and_module_specific_details", async ({
  page,
}) => {
  await page.goto("/lab/kv-cache?depth=beginner");
  await expect(page.getByTestId("depth-guide")).toContainText("历史");
  await page.getByRole("button", { name: "深入", exact: true }).click();
  await expect(page.getByTestId("depth-guide")).toContainText("2 ×");
  await page.getByRole("button", { name: "精通", exact: true }).click();
  await expect(page.getByTestId("depth-guide")).toContainText("线性");
  await page.goto("/frameworks/vllm?module=cache&depth=beginner");
  const first = await page.getByTestId("module-depth-guide").textContent();
  await page.getByRole("button", { name: "精通", exact: true }).click();
  await expect(page.getByTestId("module-depth-guide")).not.toHaveText(first!);
  await expect(page.getByTestId("module-depth-guide")).toContainText("释放");
});
test("linear_state_and_custom_shapes_are_interactive", async ({ page }) => {
  await page.goto("/models");
  await expect(page.getByTestId("delta-state")).toContainText("0.000");
  await page
    .getByRole("button", { name: "更新一个 Token", exact: true })
    .click();
  await expect(page.getByTestId("delta-state")).toContainText("0.600");
  await expect(page.getByTestId("conv-buffer")).toContainText("1.00");
  await page.getByLabel("选择模型").selectOption("teaching");
  await page.getByLabel("教学 Q 头数").selectOption("8");
  await page.getByLabel("教学 KV 头数").selectOption("1");
  await expect(page.getByTestId("tensor-shapes")).toContainText("8 × 16");
  await expect(page.getByTestId("tensor-shapes")).toContainText("1 × 16");
  await page.getByLabel("教学序列长度").fill("12");
  await expect(page.getByTestId("tensor-shapes")).toContainText("12 ×");
});
test("custom_hidden_dimension_is_integer", async ({ page }) => {
  await page.goto("/models?model=teaching");
  await page.getByLabel("隐藏维度").fill("128.5");
  await expect(page.getByLabel("隐藏维度")).toHaveValue("128");
});
test("search_finds_source_modules_and_deep_links", async ({ page }) => {
  await page.goto("/learn");
  await page.getByRole("searchbox").fill("GPUModelRunner");
  await page.getByRole("link", { name: "vLLM · GPUModelRunner" }).click();
  await expect(page).toHaveURL(/frameworks\/vllm\?module=runner/);
  await expect(page.locator(".source-inspector h2")).toHaveText(
    "GPUModelRunner",
  );
});

test("expert_formula_stays_in_bounds_with_mobile_text_zoom", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 1000 });
  await page.goto("/learn/gated-deltanet?depth=expert");
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
  ).toBeLessThanOrEqual(390);
});
