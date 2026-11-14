import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      // Editors write atomically through *.tmpdir; watching those can throw
      // EBUSY on Windows and kill the dev server.
      ignored: ["**/*.tmpdir/**", "**/.index.html.*.tmpdir/**"],
    },
  },
});
