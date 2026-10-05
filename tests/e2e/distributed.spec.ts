import { test, expect } from "@playwright/test";

test("workbench tabs retain numeric examples and describe what sharing preserves", async ({
  page,
}) => {
  await page.goto("/distributed?parallel=tp");
  await page.getByLabel("教学 GPU 数", { exact: true }).selectOption("4");
  await page.getByLabel("中间通道 F", { exact: true }).selectOption("8");
  await page
    .getByRole("button", { name: "通信原语工作台", exact: true })
    .click();
  await page
    .getByRole("button", { name: "选择 broadcast 通信", exact: true })
    .click();
  await page.getByLabel("通信 rank 数", { exact: true }).selectOption("4");
  await page.getByLabel("通信 root rank", { exact: true }).selectOption("3");
  await page
    .getByRole("button", { name: "并行切分工作台", exact: true })
    .click();
  await expect(page.getByLabel("教学 GPU 数", { exact: true })).toHaveValue(
    "4",
  );
  await expect(page.getByLabel("中间通道 F", { exact: true })).toHaveValue("8");
  await page
    .getByRole("button", { name: "通信原语工作台", exact: true })
    .click();
  await expect(page.getByLabel("通信 rank 数", { exact: true })).toHaveValue(
    "4",
  );
  await expect(page.getByLabel("通信 root rank", { exact: true })).toHaveValue(
    "3",
  );
  await expect(
    page.getByRole("button", { name: "分享主题链接", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("GPU 数等教学参数仅留在当前页。", { exact: false }),
  ).toBeVisible();
});

test("distributed workbench is navigable and preserves deep linked learning depth", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("link", { name: "多 GPU 与通信", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("多 GPU");
  await page
    .getByRole("button", { name: "通信原语工作台", exact: true })
    .click();
  await expect(page).toHaveURL(/view=collectives/);
  await expect(page.getByTestId("collective-workbench")).toBeVisible();
  await page.goto("/learn/cp?depth=expert");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("CP");
  await expect(page.getByTestId("parallel-topic-demo")).toBeVisible();
  await page.goto("/learn/all-gather?depth=advanced");
  await expect(page.getByTestId("collective-topic-demo")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("collective-topic-demo")).toBeVisible();
});

test("all six partition strategies and nine collectives retain usable shapes on phones", async ({
  page,
}) => {
  await page.setViewportSize({ width: 360, height: 900 });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/distributed?depth=advanced");
  for (const id of ["tp", "dp", "ep", "pp", "cp", "sp"]) {
    await page
      .getByRole("button", {
        name: `选择 ${id.toUpperCase()} 并行`,
        exact: true,
      })
      .click();
    await expect(page.getByTestId("parallel-explorer")).toHaveAttribute(
      "data-strategy",
      id,
    );
    await page.getByLabel("教学 GPU 数", { exact: true }).selectOption("4");
    await page
      .getByRole("group", { name: "并行演示阶段" })
      .getByRole("button")
      .last()
      .click();
    await expect(page.getByTestId("parallel-explorer")).not.toContainText(
      /NaN|undefined/,
    );
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      id,
    ).toBeLessThanOrEqual(360);
  }
  await page
    .getByRole("button", { name: "通信原语工作台", exact: true })
    .click();
  for (const id of [
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "send-recv",
  ]) {
    await page
      .getByRole("button", { name: `选择 ${id} 通信`, exact: true })
      .click();
    await expect(page.getByTestId("collective-explorer")).toHaveAttribute(
      "data-operation",
      id,
    );
    await page.getByLabel("通信 rank 数", { exact: true }).selectOption("4");
    await page
      .getByLabel("通信关键帧", { exact: true })
      .getByRole("button")
      .last()
      .click();
    await expect(page.getByTestId("collective-explorer")).not.toContainText(
      /NaN/,
    );
    await expect(page.getByLabel("通信 rank 数", { exact: true })).toHaveValue(
      "4",
    );
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      id,
    ).toBeLessThanOrEqual(360);
  }
  expect(errors).toEqual([]);
});

test("continuous partition and communication objects survive mode changes and freeze when paused", async ({
  page,
}) => {
  await page.goto("/distributed?parallel=tp");
  const board = page.getByTestId("parallel-explorer");
  await page.getByRole("button", { name: "连续演示", exact: true }).click();
  await page.getByLabel("并行播放速度", { exact: true }).selectOption("0.5");
  const packet = page.getByTestId("parallel-object").first();
  await packet.evaluate((el) => ((window as any).__parallelPacket = el));
  await page.getByRole("button", { name: "播放并行演示", exact: true }).click();
  await expect
    .poll(async () => Number(await board.getAttribute("data-position")))
    .toBeGreaterThan(0.06);
  await page.getByRole("button", { name: "暂停并行演示", exact: true }).click();
  const at = await board.getAttribute("data-position"),
    transform = await packet.getAttribute("transform");
  await page.waitForTimeout(200);
  expect(await packet.getAttribute("transform")).toBe(transform);
  await page.getByRole("button", { name: "分段演示", exact: true }).click();
  await page.getByRole("button", { name: "连续演示", exact: true }).click();
  await expect(board).toHaveAttribute("data-position", at!);
  expect(
    await packet.evaluate((el) => el === (window as any).__parallelPacket),
  ).toBe(true);
  await page
    .getByRole("button", { name: "通信原语工作台", exact: true })
    .click();
  await page.getByRole("button", { name: "连续演示通信", exact: true }).click();
  await page.getByLabel("通信动画速度", { exact: true }).selectOption("0.25");
  const communication = page.getByTestId("collective-explorer"),
    item = page.getByTestId("collective-packet").first();
  await item.evaluate((el) => ((window as any).__collectivePacket = el));
  await page.getByRole("button", { name: "播放通信动画", exact: true }).click();
  await expect
    .poll(async () => Number(await communication.getAttribute("data-position")))
    .toBeGreaterThan(0.06);
  await page.getByRole("button", { name: "暂停通信动画", exact: true }).click();
  const commAt = await communication.getAttribute("data-position"),
    location = await item.getAttribute("transform");
  await page.waitForTimeout(200);
  expect(await item.getAttribute("transform")).toBe(location);
  await page.getByLabel("通信动画速度", { exact: true }).selectOption("4");
  await page.getByRole("button", { name: "分段讲解通信", exact: true }).click();
  await page.getByRole("button", { name: "连续演示通信", exact: true }).click();
  await expect(communication).toHaveAttribute("data-position", commAt!);
  expect(
    await item.evaluate((el) => el === (window as any).__collectivePacket),
  ).toBe(true);
});

test("shape derivations accompany original numerical animations and model changes", async ({
  page,
}) => {
  await page.goto("/pipeline?depth=expert");
  const inspector = page.getByLabel("当前数据详情", { exact: true });
  await page.getByRole("button", { name: "跳到进入网络", exact: true }).click();
  await expect(inspector.getByTestId("shape-teaching")).toBeVisible();
  await expect(inspector.getByTestId("shape-teaching")).toContainText("输入");
  await page.getByLabel("选择模型", { exact: true }).selectOption("qwen36-moe");
  await page.getByRole("button", { name: "跳到进入网络", exact: true }).click();
  await expect(inspector).toContainText("2048");
  await page.goto("/learn/ffn");
  await expect(page.getByTestId("shape-teaching").first()).toBeVisible();
  await page.getByRole("button", { name: "单步原理动图", exact: true }).click();
  await expect(page.getByTestId("shape-teaching").first()).toContainText(
    "数值代入",
  );
  await page.setViewportSize({ width: 360, height: 900 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(360);
});

test("distributed deep links recover invalid values and retain text zoom and navigation at tablet widths", async ({
  page,
}) => {
  await page.goto("/distributed?view=unknown&parallel=invalid&depth=expert");
  await expect(page.getByRole("status")).toContainText("无法识别");
  await expect(page.getByTestId("parallel-explorer")).toHaveAttribute(
    "data-strategy",
    "tp",
  );
  for (const width of [390, 768, 1000, 1440, 1920]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.addStyleTag({ content: "html {font-size:20px;}" });
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
      String(width),
    ).toBeLessThanOrEqual(width);
  }
});
