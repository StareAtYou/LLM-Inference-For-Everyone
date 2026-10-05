import { test, expect } from "@playwright/test";
test("single_and_batch_journeys_finish_and_release_their_own_state", async ({
  page,
}) => {
  await page.goto("/pipeline");
  await expect(
    page.getByRole("button", { name: "批请求", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "批请求", exact: true }).click();
  await expect(page.getByTestId("request-R3")).toBeVisible();
  await page
    .getByRole("button", { name: "跳到结束与释放", exact: true })
    .click();
  for (const id of ["R1", "R2", "R3"]) {
    await expect(page.getByTestId("request-" + id)).toContainText("已完成");
    await expect(page.getByTestId("request-" + id)).toContainText("缓存 0");
  }
  await expect(page.getByTestId("trace-output")).not.toContainText("尚未产生");
  await page.getByRole("button", { name: "单请求", exact: true }).click();
  await expect(page.getByTestId("request-R3")).toHaveCount(0);
  await expect(page.getByTestId("trace-stage")).toContainText("接收");
});
test("network_nodes_open_their_own_animation_and_return_to_paused_journey", async ({
  page,
}) => {
  await page.goto("/pipeline");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.getByRole("button", { name: "展开 Embedding 原理" }).click();
  await expect(page.getByTestId("mechanism-player")).toHaveAttribute(
    "data-mechanism-id",
    "embedding",
  );
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "单步原理动图" }).click();
  await page.getByRole("button", { name: "返回全过程", exact: true }).click();
  await expect(page.getByTestId("mechanism-player")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "展开 Embedding 原理" }),
  ).toBeFocused();
  await page.getByRole("button", { name: "原理动图", exact: true }).click();
  await page.getByRole("searchbox", { name: "搜索原理动图" }).fill("RMS");
  await page.getByRole("button", { name: /打开 RMSNorm/ }).click();
  await expect(page.getByTestId("mechanism-player")).toHaveAttribute(
    "data-mechanism-id",
    "rmsnorm",
  );
});
test("topic_embeds_its_own_mechanism_and_mobile_controls_fit", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 1000 });
  await page.goto("/learn/rope");
  await expect(page.getByTestId("mechanism-player")).toHaveAttribute(
    "data-mechanism-id",
    "rope",
  );
  await page.getByRole("button", { name: "单步原理动图" }).click();
  await page.goto("/pipeline");
  await page.getByRole("button", { name: "批请求", exact: true }).click();
  await page.getByLabel("计算粒度").selectOption("operator");
  await page.getByRole("button", { name: "单步", exact: true }).click();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});

test("all_29_principles_step_to_their_outputs_without_mobile_overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/pipeline?view=mechanisms");
  await expect(page.locator(".wf-mechanism-grid button")).toHaveCount(29);
  const ids = await page
    .locator(".wf-mechanism-grid button")
    .evaluateAll((nodes) =>
      nodes.map((node) => node.getAttribute("aria-label")!),
    );
  expect(ids).toHaveLength(29);
  for (const name of ids) {
    await page.getByRole("button", { name, exact: true }).click();
    const animation = page.getByTestId("mechanism-player");
    await expect(animation).toBeVisible();
    const stages = animation.locator(".mp-stage-rail button");
    await stages.last().click();
    await expect(animation.getByTestId("mechanism-board")).not.toContainText(
      /NaN|Infinity|undefined/,
    );
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      name,
    ).toBeLessThanOrEqual(360);
    await page.getByRole("button", { name: "关闭原理动图" }).click();
  }
});

test("model_parameters_reset_playback_and_layer_jump_follows_the_real_topology", async ({
  page,
}) => {
  await page.goto("/pipeline?depth=beginner&mode=batch");
  await page.getByLabel("定位网络层").selectOption("3");
  await expect(page.getByLabel("计算粒度")).toHaveValue("operator");
  await expect(page.getByLabel("定位网络层")).toHaveValue("3");
  await expect(page.locator(".wf-group-title")).toContainText([
    "01 / 输入与调度",
    "第 4 / 64 层",
    "03 / 位置内的变换",
    "04 / 预测与生成",
  ]);
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.getByLabel("选择模型").selectOption("qwen36-moe");
  await expect(page.getByTestId("frame-position")).toHaveText(/^1 \/ \d+$/);
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  await page.getByLabel("教学输出上限").selectOption("1");
  await page.getByRole("button", { name: "跳到结束与释放" }).click();
  for (const id of ["R1", "R2", "R3"])
    await expect(page.getByTestId("request-" + id)).toContainText("输出 1");
});

test("workbench_view_returns_to_journey_when_navigation_removes_its_query", async ({
  page,
}) => {
  await page.goto("/pipeline?view=mechanisms&mode=batch");
  await expect(
    page.getByRole("button", { name: "原理动图", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("link", { name: "推理流程", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "完整推理", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("request-R3")).toHaveCount(0);
});

test("layer_jump_preserves_the_decode_round_and_release_labels_stop_feedback", async ({
  page,
}) => {
  await page.goto("/pipeline?depth=beginner");
  await page.getByRole("button", { name: "下一轮" }).click();
  await page.getByRole("button", { name: "下一轮" }).click();
  await expect(page.locator(".wf-board-toolbar")).toContainText("ROUND 02");
  await page.getByLabel("定位网络层").selectOption("3");
  await expect(page.locator(".wf-board-toolbar")).toContainText("ROUND 02");
  await page.getByRole("button", { name: "跳到结束与释放" }).click();
  await expect(page.getByLabel("当前数据传递")).toContainText("释放状态");
  await expect(page.getByLabel("当前数据传递")).not.toContainText("回送下一轮");
});
