import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Vite 配置：/api 开头的请求代理到 FastAPI 后端（8000），前端代码里直接 fetch('/api/...') 即可
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
