import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The console is built to `web/dist`, which the FastAPI service mounts at "/".
// `dist` is committed deliberately (see web/README.md): the service is then
// runnable with Python alone, and a reviewer needs no Node toolchain to see
// the UI. `npm run build` regenerates it.
//
// In `npm run dev`, Vite serves the UI on :5173 and proxies the two API routes
// to a uvicorn process on :8000, so the front end always talks to the real
// service over real HTTP — never to a mock.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/ask': 'http://127.0.0.1:8000',
      '/healthz': 'http://127.0.0.1:8000',
    },
  },
})
