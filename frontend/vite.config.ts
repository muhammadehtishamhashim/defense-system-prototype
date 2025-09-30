import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react(), tailwindcss()],
    base: './',
    server: {
      host: true,
      port: 3000,
    },
    preview: {
      port: 4173,
      host: true,
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: mode === 'development', // Enable sourcemaps only in development
      minify: mode === 'production' ? 'esbuild' : false,
      target: 'es2015', // Ensure compatibility with older browsers
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'react-router-dom'],
            ui: ['@headlessui/react', '@heroicons/react'],
            charts: ['recharts'],
            utils: ['axios', 'clsx', 'class-variance-authority'],
          },
          // Ensure consistent file naming
          chunkFileNames: 'assets/[name]-[hash].js',
          entryFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]',
        },
      },
      // Increase chunk size warning limit
      chunkSizeWarningLimit: 1000,
    },
    define: {
      // Ensure environment variables are properly defined for production
      __DEV__: JSON.stringify(mode === 'development'),
      'process.env.NODE_ENV': JSON.stringify(mode),
      // Make sure VITE_ prefixed env vars are available
      'import.meta.env.VITE_API_URL': JSON.stringify(env.VITE_API_URL || 'http://localhost:8000'),
      'import.meta.env.VITE_NODE_ENV': JSON.stringify(env.VITE_NODE_ENV || mode),
    },
    // Ensure proper resolution for production builds
    resolve: {
      alias: {
        '@': '/src',
      },
    },
    // Optimize dependencies
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-router-dom', 'axios'],
    },
  }
})
