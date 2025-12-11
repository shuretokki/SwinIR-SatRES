import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Components from 'unplugin-vue-components/vite'
import AutoImport from 'unplugin-auto-import/vite'
import Icons from 'unplugin-icons/vite'
import IconsResolver from 'unplugin-icons/resolver'
import { viteStaticCopy } from 'vite-plugin-static-copy'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
    Icons({
      autoInstall: true,
      compiler: 'vue3',
    }),
    Components({
      dts: true,
      resolvers: [
        IconsResolver({
          prefix: 'icon',
          enabledCollections: ['solar'],
        }),
      ],
    }),
    AutoImport({
      imports: ['vue', '@vueuse/core'],
      dts: true,
      vueTemplate: true,
    }),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'onnx-resources'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/*.mjs',
          dest: 'onnx-resources'
        }
      ]
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
})
