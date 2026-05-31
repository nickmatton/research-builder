import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// FastAPI runs on :7777; Vite dev server on :5173.
// /api and /ws are proxied so the same-origin fetch + WebSocket code
// in the client works identically in dev and prod (Phase 5 will serve
// the built bundle from FastAPI itself, collapsing both to :7777).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:7777", changeOrigin: true },
      "/ws": {
        target: "ws://127.0.0.1:7777",
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
