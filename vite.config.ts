import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig(({ mode }) => ({
  base:
    mode === "pages"
      ? process.env.PAGES_BASE_PATH || "/LLM-Inference-For-Everyone/"
      : "/",
  plugins: [react()],
  test: { include: ["tests/unit/**/*.test.ts"] },
}));
