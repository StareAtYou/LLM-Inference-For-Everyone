import { afterAll, beforeAll, beforeEach, expect, it } from "vitest";
import { chromium, type Browser, type Page } from "@playwright/test";
import { createServer, type ViteDevServer } from "vite";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

let server: ViteDevServer;
let browser: Browser;
let page: Page;
let cacheDirectory: string;
let browserErrors: string[] = [];
const harness = `
import React from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter } from 'react-router';
import { PreferencesProvider } from '/src/hooks/usePreferences.ts';
import ParallelExplorer from '/src/distributed/ParallelExplorer.tsx';
import '/src/styles/tokens.css';
import '/src/styles/global.css';
createRoot(document.getElementById('root')).render(
  React.createElement(HashRouter, null,
    React.createElement(PreferencesProvider, null,
      React.createElement(ParallelExplorer))));
`;

beforeAll(async () => {
  cacheDirectory = await mkdtemp(join(tmpdir(), "parallel-browser-cache-"));
  server = await createServer({
    configFile: false,
    cacheDir: cacheDirectory,
    optimizeDeps: {
      include: [
        "react",
        "react-dom/client",
        "react-router",
        "react/jsx-runtime",
      ],
    },
    server: { host: "127.0.0.1", port: 0 },
    plugins: [
      {
        name: "parallel-browser-fixture",
        resolveId(id) {
          if (id === "/__parallel.js") return "\0parallel-harness";
        },
        load(id) {
          if (id === "\0parallel-harness") return harness;
        },
        configureServer(vite) {
          vite.middlewares.use((request, response, next) => {
            if (request.url !== "/__parallel.html") return next();
            response.setHeader("Content-Type", "text/html");
            response.end(
              '<meta name="viewport" content="width=device-width, initial-scale=1"><div id="root"></div><script type="module" src="/__parallel.js"></script>',
            );
          });
        },
      },
    ],
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.platform === "darwin" ? "chrome" : "chromium",
  });
  page = await browser.newPage({ viewport: { width: 1200, height: 1100 } });
  page.setDefaultTimeout(8_000);
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("requestfailed", (request) =>
    browserErrors.push(
      `${request.method()} ${request.url()}: ${request.failure()?.errorText}`,
    ),
  );
}, 20_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (cacheDirectory)
    await rm(cacheDirectory, { recursive: true, force: true });
});

beforeEach(async () => {
  browserErrors = [];
  await page.setViewportSize({ width: 1200, height: 1100 });
  const address = server.httpServer!.address() as { port: number };
  await page.goto(
    `http://127.0.0.1:${address.port}/__parallel.html#/distributed?depth=advanced`,
  );
  try {
    await page.getByTestId("parallel-explorer").waitFor({ state: "visible" });
  } catch (error) {
    throw new Error(
      `Parallel fixture did not render: ${browserErrors.join("; ")}\n${String(error)}`,
    );
  }
  expect(browserErrors).toEqual([]);
}, 20_000);

it("all six strategies continuously interpolate persistent objects and freeze when paused", async () => {
  for (const id of ["tp", "dp", "ep", "pp", "cp", "sp"]) {
    await page
      .getByRole("button", {
        name: `选择 ${id.toUpperCase()} 并行`,
        exact: true,
      })
      .click();
    const surface = page.getByTestId("parallel-explorer");
    await surface
      .getByRole("button", { name: "连续演示", exact: true })
      .click();
    await surface
      .getByRole("button", { name: "重置并行演示", exact: true })
      .click();
    await page.evaluate(() => {
      (window as any).parallelRefs = Array.from(
        document.querySelectorAll('[data-testid="parallel-object"]'),
      );
    });
    await surface
      .getByRole("button", { name: "播放并行演示", exact: true })
      .click();
    await page.waitForFunction(
      () =>
        Number(
          document
            .querySelector('[data-testid="parallel-explorer"]')
            ?.getAttribute("data-position"),
        ) > 0.08,
    );
    await surface
      .getByRole("button", { name: "暂停并行演示", exact: true })
      .click();
    const paused = await surface.getAttribute("data-position");
    const transforms = await surface
      .getByTestId("parallel-object")
      .evaluateAll((objects) =>
        objects.map((object) => object.getAttribute("transform")),
      );
    await page.waitForTimeout(130);
    expect(await surface.getAttribute("data-position")).toBe(paused);
    expect(
      await surface
        .getByTestId("parallel-object")
        .evaluateAll((objects) =>
          objects.map((object) => object.getAttribute("transform")),
        ),
    ).toEqual(transforms);
    await surface
      .getByRole("combobox", { name: "并行播放速度", exact: true })
      .selectOption("2");
    await surface
      .getByRole("button", { name: "分段演示", exact: true })
      .click();
    await surface
      .getByRole("button", { name: "并行单步", exact: true })
      .click();
    expect(
      await page.evaluate(() =>
        (window as any).parallelRefs.every(
          (object: Element, index: number) =>
            object ===
            document.querySelectorAll('[data-testid="parallel-object"]')[index],
        ),
      ),
    ).toBe(true);
    expect(await surface.getAttribute("data-strategy")).toBe(id);
    expect(await page.evaluate(() => location.hash)).toContain(
      `parallel=${id}`,
    );
    await surface
      .getByRole("button", { name: "重置并行演示", exact: true })
      .click();
    expect(await surface.getAttribute("data-position")).toBe("0.0000");
  }
  expect(browserErrors).toEqual([]);
}, 30_000);

it("preserves per-strategy parameters and keeps detailed matrices in local scroll areas on mobile", async () => {
  const surface = page.getByTestId("parallel-explorer");
  await surface
    .getByRole("button", { name: "选择 TP 并行", exact: true })
    .click();
  await surface
    .getByRole("combobox", { name: "中间通道 F", exact: true })
    .selectOption("8");
  await surface
    .getByRole("button", { name: "选择 EP 并行", exact: true })
    .click();
  await surface
    .getByRole("combobox", { name: "教学 GPU 数", exact: true })
    .selectOption("4");
  await surface
    .getByRole("combobox", { name: "Token 数 T", exact: true })
    .selectOption("8");
  await surface
    .getByRole("combobox", { name: "路由 top-k", exact: true })
    .selectOption("1");
  await surface
    .getByRole("button", { name: "选择 TP 并行", exact: true })
    .click();
  expect(
    await surface
      .getByRole("combobox", { name: "中间通道 F", exact: true })
      .inputValue(),
  ).toBe("8");
  await surface
    .getByRole("button", { name: "选择 EP 并行", exact: true })
    .click();
  expect(
    await surface
      .getByRole("combobox", { name: "路由 top-k", exact: true })
      .inputValue(),
  ).toBe("1");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  expect(
    await surface
      .locator(".px-scene-scroll")
      .evaluate((element) => element.scrollWidth > element.clientWidth),
  ).toBe(true);
  await page.evaluate(() => {
    document.documentElement.style.fontSize = "200%";
  });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: "/tmp/parallel-mobile-ep.png",
    fullPage: true,
  });
  expect(browserErrors).toEqual([]);
}, 15_000);
