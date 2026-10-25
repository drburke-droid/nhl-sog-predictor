import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` writes the static site to docs/traffic/ (served by GitHub Pages with the rest of docs/)
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../../../docs/traffic", emptyOutDir: true },
});
