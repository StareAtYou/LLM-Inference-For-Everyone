import { afterAll, beforeAll, expect, test } from "vitest";
import { chromium, type Browser, type Page } from "@playwright/test";
import { createServer, type ViteDevServer } from "vite";

let server: ViteDevServer;
let browser: Browser;
let page: Page;
const waitFor = (predicate: () => unknown) =>
  page.waitForFunction(predicate, undefined, { polling: 10, timeout: 5000 });

// Use real React and DOM lifecycle; control only RAF's external clock.
const harness = `
import React, { useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { useTimeline } from '/src/hooks/useTimeline.ts';
let timestamp = 0, requestId = 0;
const frames = new Map();
Object.defineProperty(performance, 'now', { value: () => timestamp });
window.requestAnimationFrame = callback => {
  frames.set(++requestId, callback);
  return requestId;
};
window.cancelAnimationFrame = id => frames.delete(id);
window.tick = time => {
  timestamp = time;
  const pending = [...frames.values()];
  frames.clear();
  pending.forEach(callback => callback(timestamp));
};
window.outputs = {};
window.model = { resetKey: 'first', frameCount: 4, intervalMs: 100 };
function Surface({ name }) {
  const ref = useRef(null);
  const timeline = useTimeline({ ...window.model, surfaceRef: ref });
  window.outputs[name] = timeline;
  return React.createElement('div', { ref, id: name, style: { height: 100 } }, name);
}
const root = createRoot(document.getElementById('root'));
window.render = () => root.render(React.createElement(React.Fragment, null,
  React.createElement(Surface, { name: 'a' }),
  React.createElement(Surface, { name: 'b' })));
window.unmount = () => root.unmount();
window.render();
`;

beforeAll(async () => {
  server = await createServer({
    configFile: false,
    server: { host: "127.0.0.1", port: 0 },
    plugins: [
      {
        name: "timeline-hook-test-harness",
        resolveId(id) {
          if (id === "/__timeline-harness.js") return "\0timeline-harness";
        },
        load(id) {
          if (id === "\0timeline-harness") return harness;
        },
        configureServer(vite) {
          vite.middlewares.use((req, res, next) => {
            if (req.url !== "/__timeline-harness.html") return next();
            res.setHeader("Content-Type", "text/html");
            res.end(
              '<div id="root"></div><script type="module" src="/__timeline-harness.js"></script>',
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
  page = await browser.newPage();
}, 20_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
});

test("hook keeps the exact playhead across mode, speed, duration, and model changes", async () => {
  const address = server.httpServer!.address() as { port: number };
  await page.goto(`http://127.0.0.1:${address.port}/__timeline-harness.html`);
  await waitFor(() => Boolean((window as any).outputs?.a));
  await page.evaluate(() => {
    const t = (window as any).outputs.a;
    t.setMode("continuous");
    t.play();
  });
  await waitFor(() => (window as any).outputs.a.playing);
  await page.evaluate(() => (window as any).tick(25));
  await waitFor(() => (window as any).outputs.a.position === 0.25);
  await page.evaluate(() => (window as any).outputs.a.setSpeed(2));
  await waitFor(() => (window as any).outputs.a.speed === 2);
  expect(await page.evaluate(() => (window as any).outputs.a.position)).toBe(
    0.25,
  );
  await page.evaluate(() => (window as any).tick(50));
  await waitFor(() => (window as any).outputs.a.position === 0.75);
  await page.evaluate(() => (window as any).outputs.a.setMode("staged"));
  await waitFor(() => (window as any).outputs.a.mode === "staged");
  expect(
    await page.evaluate(() => ({
      position: (window as any).outputs.a.position,
      playing: (window as any).outputs.a.playing,
    })),
  ).toEqual({ position: 0.75, playing: false });
  await page.evaluate(() => {
    (window as any).model.intervalMs = 200;
    (window as any).render();
  });
  await waitFor(() => (window as any).outputs.a.durationMs === 600);
  expect(await page.evaluate(() => (window as any).outputs.a.position)).toBe(
    0.75,
  );
  await page.evaluate(() => {
    (window as any).model.resetKey = "second";
    (window as any).render();
  });
  await waitFor(() => (window as any).outputs.a.position === 0);
  expect(
    await page.evaluate(() => ({
      mode: (window as any).outputs.a.mode,
      speed: (window as any).outputs.a.speed,
      playing: (window as any).outputs.a.playing,
    })),
  ).toEqual({ mode: "staged", speed: 2, playing: false });
}, 15_000);

test("each hook pauses only its own offscreen surface, then visibility pauses both", async () => {
  const address = server.httpServer!.address() as { port: number };
  await page.goto(`http://127.0.0.1:${address.port}/__timeline-harness.html`);
  await waitFor(() => Boolean((window as any).outputs?.b));
  await page.evaluate(() => {
    for (const name of ["a", "b"]) {
      (window as any).outputs[name].setMode("continuous");
      (window as any).outputs[name].play();
    }
  });
  await waitFor(
    () =>
      (window as any).outputs.a.playing && (window as any).outputs.b.playing,
  );
  await page.evaluate(
    () => (document.getElementById("a")!.style.display = "none"),
  );
  await waitFor(() => !(window as any).outputs.a.playing);
  expect(await page.evaluate(() => (window as any).outputs.b.playing)).toBe(
    true,
  );
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {
      configurable: true,
      value: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await waitFor(() => !(window as any).outputs.b.playing);
  await page.evaluate(() => {
    (window as any).outputs.b.play();
    (window as any).tick(100);
  });
  expect(await page.evaluate(() => (window as any).outputs.b.playing)).toBe(
    false,
  );
  await page.evaluate(() => (window as any).unmount());
}, 15_000);
