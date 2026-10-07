import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In docker the backend is a service name, not localhost. Set VITE_API_TARGET
// to override; vite.config runs on the node side, so this stays a build-time
// value rather than something the client has to know about.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  // vitest reads its settings from here
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
