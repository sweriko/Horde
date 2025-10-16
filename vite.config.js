import { defineConfig } from 'vite'

export default defineConfig({
  root: '.',
  server: {
    port: 3000,
    open: true,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
  },
  optimizeDeps: {
    include: [
      'three',
      '@dimforge/rapier3d-compat',
      'lil-gui',
    ],
  },
  assetsInclude: ['**/*.gltf', '**/*.glb', '**/*.fbx', '**/*.obj', '**/*.wasm'],
})