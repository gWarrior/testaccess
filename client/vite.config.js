import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev, proxy API/content calls to the NestJS server (default port 3001).
// In production the built files are served by NestJS itself, same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:3001',
      '/content': 'http://localhost:3001',
      '/health': 'http://localhost:3001',
    },
  },
  build: {
    outDir: 'dist',
  },
});
