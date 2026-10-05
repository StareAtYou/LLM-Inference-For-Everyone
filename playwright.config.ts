import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30000,
  expect: { timeout: 5000 },
  use: {
    baseURL: process.env.PREVIEW_URL || "http://127.0.0.1:5173",
    headless: true,
    channel:
      process.env.PLAYWRIGHT_CHANNEL ||
      (process.platform === "darwin" ? "chrome" : "chromium"),
    screenshot: "only-on-failure",
  },
  webServer: process.env.PREVIEW_URL
    ? undefined
    : {
        command: "npm run dev -- --port 5173",
        url: "http://127.0.0.1:5173",
        reuseExistingServer: true,
      },
  reporter: "list",
});
