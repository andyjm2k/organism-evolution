import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./",
  build: { sourcemap: false, target: "es2022" },
  server: { host: "127.0.0.1", port: 5173 },
  test: { environment: "node", include: ["test/**/*.test.ts"] },
});
