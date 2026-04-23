import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    // Each test file gets its own worker so vi.mock hoisting is isolated.
    pool: "forks",
  },
});
