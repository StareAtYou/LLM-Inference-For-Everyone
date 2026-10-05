import { test, expect } from "@playwright/test";

test("continuous_journey_moves_between_keyframes_and_preserves_the_paused_playhead", async ({
  page,
}) => {
  await page.goto("/pipeline?depth=beginner");
  await page
    .getByRole("button", { name: "连续演示全过程", exact: true })
    .click();
  const board = page.getByLabel("全过程数据流", { exact: true });
  await expect(board).toHaveAttribute("data-mode", "continuous");
  await expect(page.getByTestId("continuous-journey-scene")).toBeVisible();
  await page.getByLabel("播放速度").selectOption("0.25");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await expect
    .poll(async () => Number(await board.getAttribute("data-position")))
    .toBeGreaterThan(0.05);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  const at = Number(await board.getAttribute("data-position"));
  expect(at % 1).toBeGreaterThan(0);
  expect(at).toBeLessThan(1);
  const particle = page.locator(".cj-packet").first();
  const frozen = await particle.getAttribute("transform");
  await page.waitForTimeout(250);
  expect(await particle.getAttribute("transform")).toBe(frozen);
  await page.getByLabel("播放速度").selectOption("4");
  expect(Number(await board.getAttribute("data-position"))).toBe(at);
  await page
    .getByRole("button", { name: "分段讲解全过程", exact: true })
    .click();
  expect(Number(await board.getAttribute("data-position"))).toBe(at);
  await page
    .getByRole("button", { name: "连续演示全过程", exact: true })
    .click();
  expect(Number(await board.getAttribute("data-position"))).toBe(at);
  await page.getByRole("button", { name: "跳到结束与释放" }).click();
  await expect(page.getByTestId("trace-stage")).toContainText("释放");
  await expect(page.getByTestId("request-R1")).toContainText("缓存 0");
});

test("principle_continuous_mode_uses_the_same_keyframes_and_reset_stops_the_local_clock", async ({
  page,
}) => {
  await page.goto("/learn/rope");
  const player = page.getByTestId("mechanism-player");
  await page.getByRole("button", { name: "单步原理动图" }).click();
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  await expect(player).toHaveAttribute("data-position", "1");
  await expect(page.getByTestId("continuous-mechanism-scene")).toBeVisible();
  await page.getByLabel("原理动画速度").selectOption("0.25");
  await page.getByRole("button", { name: "播放原理动图" }).click();
  await expect
    .poll(async () => Number(await player.getAttribute("data-position")))
    .toBeGreaterThan(1.05);
  await page.getByRole("button", { name: "暂停原理动图" }).click();
  const at = await player.getAttribute("data-position");
  await page.getByRole("button", { name: "分段讲解原理", exact: true }).click();
  await expect(player).toHaveAttribute("data-position", at!);
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  await page.getByRole("button", { name: "重置原理动图" }).click();
  await expect(player).toHaveAttribute("data-position", "0");
  await expect(
    page.getByRole("button", { name: "播放原理动图" }),
  ).toBeVisible();
});

test("all_29_continuous_principles_fit_mobile_and_show_finite_calculated_results", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 1000 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/pipeline?view=mechanisms");
  await expect(page.locator(".wf-mechanism-grid button")).toHaveCount(29);
  const names = await page
    .locator(".wf-mechanism-grid button")
    .evaluateAll((nodes) => nodes.map((n) => n.getAttribute("aria-label")!));
  for (const name of names) {
    await page.getByRole("button", { name, exact: true }).click();
    await page
      .getByRole("button", { name: "连续演示原理", exact: true })
      .click();
    await expect(page.getByTestId("continuous-mechanism-scene")).toBeVisible();
    await page
      .getByTestId("mechanism-player")
      .locator(".mp-stage-rail button")
      .last()
      .click();
    await expect(page.getByTestId("mechanism-player")).not.toContainText(
      /NaN|Infinity|undefined/,
    );
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      name,
    ).toBeLessThanOrEqual(360);
    await page.getByRole("button", { name: "关闭原理动图" }).click();
  }
});

test("journey_mounted_after_the_library_still_pauses_offscreen", async ({
  page,
}) => {
  await page.goto("/pipeline?view=mechanisms");
  await page.getByRole("button", { name: "完整推理", exact: true }).click();
  await page
    .getByRole("button", { name: "连续演示全过程", exact: true })
    .click();
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await page.locator(".site-footer").scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("button", { name: "播放", exact: true }),
  ).toBeVisible();
  const board = page.getByLabel("全过程数据流", { exact: true });
  const position = await board.getAttribute("data-position");
  await page.waitForTimeout(250);
  await expect(board).toHaveAttribute("data-position", position!);
});

test("continuous_cuda_graph_snapshot_has_no_independent_fixed_speed_animation", async ({
  page,
}) => {
  await page.goto("/learn/cuda-graphs");
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  await page.getByLabel("原理动画速度").selectOption("0.25");
  await page.getByRole("button", { name: "播放原理动图" }).click();
  const legacy = page.locator(".mp-graph-edge circle").first();
  await expect(legacy).toHaveCount(1);
  expect(await legacy.evaluate((e) => getComputedStyle(e).animationName)).toBe(
    "none",
  );
});

test("continuous_packets_keep_their_DOM_identity_across_operators_and_layers", async ({
  page,
}) => {
  await page.goto("/pipeline?mode=batch");
  await page
    .getByRole("button", { name: "连续演示全过程", exact: true })
    .click();
  const packets = page.locator(".cj-packet");
  await expect(packets).toHaveCount(3);
  await packets.evaluateAll((nodes) =>
    nodes.forEach((n, i) => n.setAttribute("data-identity-probe", String(i))),
  );
  await page.getByLabel("播放速度").selectOption("4");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await expect
    .poll(async () =>
      Number(
        await page
          .getByLabel("全过程数据流", { exact: true })
          .getAttribute("data-position"),
      ),
    )
    .toBeGreaterThan(14);
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  expect(
    await packets.evaluateAll((nodes) =>
      nodes.map((n) => n.getAttribute("data-identity-probe")),
    ),
  ).toEqual(["0", "1", "2"]);
  for (const packet of await packets.all())
    expect(await packet.getAttribute("transform")).toMatch(
      /translate\([\d.]+ [\d.]+\)/,
    );
  await expect(page.getByTestId("inference-map")).toBeHidden();
  expect(await page.locator(".cj-module").count()).toBeGreaterThan(15);
});

test("continuous_rope_rotates_during_playback_without_remounting_the_data_objects", async ({
  page,
}) => {
  await page.goto("/learn/rope");
  await page.getByRole("button", { name: "单步原理动图" }).click();
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  const objects = page.locator("[data-motion-object]");
  await objects.evaluateAll((nodes) =>
    nodes.forEach((n) => n.setAttribute("data-identity-probe", "same")),
  );
  const rotation = page.locator("[data-rotation-angle]");
  const before = Number(await rotation.getAttribute("data-rotation-angle"));
  await page.getByRole("button", { name: "播放原理动图", exact: true }).click();
  await expect
    .poll(async () =>
      Number(await rotation.getAttribute("data-rotation-angle")),
    )
    .toBeGreaterThan(before + 0.03);
  await page.getByRole("button", { name: "暂停原理动图", exact: true }).click();
  expect(
    await objects.evaluateAll((nodes) =>
      nodes.every((n) => n.getAttribute("data-identity-probe") === "same"),
    ),
  ).toBe(true);
  const paused = await rotation.getAttribute("data-rotation-angle");
  await page.waitForTimeout(150);
  await expect(rotation).toHaveAttribute("data-rotation-angle", paused!);
});

test("phone_principle_labels_remain_readable_inside_a_local_scrollable_canvas", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 900 });
  await page.goto("/learn/rope");
  await page.getByRole("button", { name: "连续演示原理", exact: true }).click();
  const viewport = page.getByRole("region", {
    name: "连续原理通路，可横向滚动",
  });
  await expect(viewport).toBeVisible();
  const pixels = await page
    .locator(".cm-zone-title")
    .first()
    .evaluate(
      (node) =>
        parseFloat(getComputedStyle(node).fontSize) *
        (node as SVGTextElement).getScreenCTM()!.a,
    );
  expect(pixels).toBeGreaterThanOrEqual(11);
  expect(await viewport.evaluate((n) => n.scrollWidth > n.clientWidth)).toBe(
    true,
  );
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(360);
});
