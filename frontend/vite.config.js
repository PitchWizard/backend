// vite.config.js (또는 ts/mts)
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',  // 또는 '127.0.0.1'
    port: 3000,
    strictPort: true, // 5173 아니면 안 띄우게 (다른 포트로 자동 변경 방지)
  },
})
