import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/pages",
  use: {
    baseURL:
      process.env.PAGES_URL ||
      "http://127.0.0.1:4180/LLM-Inference-For-Everyone/",
    channel: process.platform === "darwin" ? "chrome" : "chromium",
  },
  webServer: process.env.PAGES_URL
    ? undefined
    : {
        command: "node scripts/serve-pages-preview.mjs",
        url: "http://127.0.0.1:4180/LLM-Inference-For-Everyone/",
      },
});
