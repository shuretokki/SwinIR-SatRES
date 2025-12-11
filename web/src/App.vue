<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useTitle } from '@vueuse/core'
import { checkHealth } from './api'
import ComparisonSlider from './components/ComparisonSlider.vue'

useTitle('SwinIR - Replicate Dashboard')
const currentView = ref('dashboard') // 'dashboard' | 'model'
const serverStatus = ref('offline')
const gpuName = ref('')
const isSidebarOpen = ref(true)

const searchQuery = ref('')
const showApiModal = ref(false)

const models = ref([
  {
    id: 'vdrone-swinir-x4',
    name: 'vdrone-swinir-x4',
    author: 'shuretokki / swinir',
    description: 'Restores high-quality images from low-quality inputs. Optimized for generic real-world images with 4x upscaling.',
    public: true
  }
])

const filteredModels = computed(() => {
  if (!searchQuery.value) return models.value
  const query = searchQuery.value.toLowerCase()
  return models.value.filter(m =>
    m.name.toLowerCase().includes(query) ||
    m.description.toLowerCase().includes(query) ||
    m.author.toLowerCase().includes(query)
  )
})

const inputImage = ref<string | null>(null)
const outputImage = ref<string | null>(null)
const inputImageFile = ref<File | null>(null)
const isProcessing = ref(false)
const progress = ref(0)
const progressStage = ref('')
const inferenceTime = ref(0)
const fileInput = ref<HTMLInputElement | null>(null)
const ws = ref<WebSocket | null>(null)

interface HistoryItem {
  id: number
  input: string
  output: string
  timestamp: number
}
const history = ref<HistoryItem[]>([])


onMounted(async () => {
  try {
    const health = await checkHealth()
    if (health.status === 'healthy') {
      serverStatus.value = 'online'
      gpuName.value = health.provider === 'CUDAExecutionProvider' ? 'NVIDIA GPU' : 'CPU'
    } else {
      serverStatus.value = 'offline'
    }
  } catch (e) {
    serverStatus.value = 'offline'
  }
})


const handleFile = (file: File) => {
  if (!file || !file.type.startsWith('image/')) return

  const reader = new FileReader()
  reader.onload = (e) => {
    inputImage.value = e.target?.result as string
    outputImage.value = null
    inputImageFile.value = file
  }
  reader.readAsDataURL(file)
}

const onDrop = (e: DragEvent) => {
  const file = e.dataTransfer?.files[0]
  if (file) {
    handleFile(file)
  }
}

const onFileChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    handleFile(file)
  }
}

const runInference = () => {
  if (!inputImageFile.value) return
  isProcessing.value = true
  progress.value = 0
  progressStage.value = 'Connecting...'

  ws.value = new WebSocket('ws://localhost:8000/ws')

  ws.value.onopen = () => {
    ws.value?.send(JSON.stringify({
      type: 'infer',
      image: inputImage.value,
      max_size: 512
    }))
  }

  ws.value.onmessage = (event) => {
    const data = JSON.parse(event.data)

    if (data.type === 'progress') {
      progress.value = data.progress * 100
      progressStage.value = data.message
    } else if (data.type === 'result') {
      outputImage.value = data.image
      inferenceTime.value = data.inference_time
      isProcessing.value = false


      if (inputImage.value && data.image) {
        history.value.unshift({
          id: Date.now(),
          input: inputImage.value,
          output: data.image,
          timestamp: Date.now()
        })
      }

      ws.value?.close()
    } else if (data.type === 'error') {
      alert(`Error: ${data.message}`)
      isProcessing.value = false
      ws.value?.close()
    }
  }

  ws.value.onerror = () => {
    alert('WebSocket connection error')
    isProcessing.value = false
  }
}

const downloadResult = () => {
  if (!outputImage.value) return
  const a = document.createElement('a')
  a.href = outputImage.value
  a.download = `swinir-${Date.now()}.png`
  a.click()
}

const triggerRunOrUpload = () => {
    if (!inputImage.value) {
        fileInput.value?.click()
    } else {
        runInference()
    }
}
</script>

<template>
  <div class="flex h-screen w-full bg-replicate-bg text-replicate-text overflow-hidden">


    <aside class="w-64 flex-shrink-0 border-r border-replicate-border bg-[#0f0f0f] flex flex-col justify-between transition-all duration-300 transform"
           :class="{ '-translate-x-full absolute z-50 h-full': !isSidebarOpen, 'static translate-x-0': isSidebarOpen }">


      <div class="h-16 flex items-center px-6 border-b border-replicate-border">
        <div class="flex items-center gap-3 cursor-pointer" @click="currentView = 'dashboard'">
          <div class="w-8 h-8 bg-white flex items-center justify-center font-bold text-black text-xl tracking-tighter">
            <svg viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6"><path d="M2 2h20v20H2V2zm4 4v12h4V6H6zm8 0v12h4V6h-4z"/></svg>
          </div>
          <span class="font-bold text-lg tracking-tight">Replicate</span>
        </div>
      </div>


      <nav class="flex-1 p-4 space-y-1">
        <a href="#" @click.prevent="currentView = 'dashboard'"
           class="flex items-center gap-3 px-3 py-2 transition-colors group"
           :class="currentView === 'dashboard' ? 'bg-replicate-hover text-white' : 'text-replicate-muted hover:text-white'">
          <icon-solar-widget-linear class="w-5 h-5 opacity-70 group-hover:opacity-100" />
          <span class="font-medium text-sm">Dashboard</span>
        </a>
        <a href="#" @click.prevent="currentView = 'gallery'"
           class="flex items-center gap-3 px-3 py-2 text-replicate-muted hover:text-white hover:bg-replicate-hover transition-colors group"
           :class="currentView === 'gallery' ? 'bg-replicate-hover text-white' : ''">
          <icon-solar-gallery-linear class="w-5 h-5 opacity-70 group-hover:opacity-100" />
          <span class="font-medium text-sm">Gallery</span>
        </a>
      </nav>



    </aside>


    <div class="flex-1 flex flex-col min-w-0 bg-replicate-bg">


      <header class="h-16 border-b border-replicate-border flex items-center justify-between px-6 bg-[#0f0f0f]/50 backdrop-blur-md sticky top-0 z-40">


        <div class="flex-1 max-w-xl relative group">
           <icon-solar-magnifer-linear class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 group-focus-within:text-white transition-colors" />
           <input type="text" placeholder="Search models..." v-model="searchQuery"
                  class="w-full bg-[#1a1a1a] border border-replicate-border rounded-md py-1.5 pl-10 pr-4 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-gray-500 focus:ring-1 focus:ring-gray-500 transition-all" />
           <div class="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-gray-600 border border-gray-700 rounded px-1.5 py-0.5">/</div>
        </div>


        <div class="flex items-center gap-2 ml-6">
           <button @click="currentView = 'docs'"
                   class="px-3 py-1.5 text-xs font-medium rounded-none transition-all border border-transparent"
                   :class="currentView === 'docs' ? 'bg-white text-black border-white shadow-lg shadow-white/10' : 'text-gray-400 hover:text-white hover:border-gray-800'">
             Docs
           </button>
           <button @click="currentView = 'support'"
                   class="px-3 py-1.5 text-xs font-medium rounded-none transition-all border border-transparent"
                   :class="currentView === 'support' ? 'bg-white text-black border-white shadow-lg shadow-white/10' : 'text-gray-400 hover:text-white hover:border-gray-800'">
             Support
           </button>
        </div>
      </header>


      <main v-if="currentView === 'dashboard'" class="flex-1 overflow-y-auto p-8 animate-fade-in custom-scrollbar">
        <h1 class="text-2xl font-bold text-white mb-6">Dashboard</h1>


        <div class="relative overflow-hidden bg-gradient-to-br from-[#1a1a1a] via-[#000] to-[#1a1a1a] border border-replicate-border p-10 text-white mb-10 group">
          <div class="relative z-10 max-w-2xl">
            <h2 class="text-5xl font-black mb-4 tracking-tighter uppercase leading-none">
               Super<br/>Resolution
            </h2>
            <p class="text-gray-400 mb-8 text-lg font-light max-w-md leading-relaxed border-l-2 border-orange-500 pl-4">
               Restore details. Enhance quality. Powered by Swin Transformer architecture.
            </p>
            <button @click="currentView = 'model'" class="px-8 py-3 bg-white text-black font-bold hover:bg-gray-200 shadow-[0_0_30px_rgba(255,255,255,0.1)] hover:shadow-[0_0_40px_rgba(255,255,255,0.3)] transition-all flex items-center gap-2 group-hover:translate-x-1">
              Start Upscaling
              <icon-solar-rocket-2-linear class="w-4 h-4" />
            </button>
          </div>

          <div class="absolute right-0 top-0 bottom-0 w-1/2 opacity-20 bg-[radial-gradient(circle_at_center,_white_1px,_transparent_1px)] bg-[length:20px_20px]"></div>
          <div class="absolute -right-20 -bottom-40 w-96 h-96 bg-orange-500/20 blur-[100px] rounded-full pointer-events-none"></div>
        </div>

        <h3 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Recent Models</h3>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">

          <div v-for="model in filteredModels" :key="model.id" @click="currentView = 'model'" class="group bg-replicate-card border border-replicate-border rounded-xl p-5 hover:border-gray-600 transition-all cursor-pointer relative overflow-hidden">
            <div class="flex items-start justify-between mb-4">
              <div class="flex items-center gap-3">
                <div>
                  <h4 class="font-semibold text-white group-hover:text-blue-400 transition-colors">{{ model.name }}</h4>
                  <p class="text-xs text-replicate-muted">{{ model.author }}</p>
                </div>
              </div>
            </div>
            <p class="text-sm text-gray-400 mb-4 line-clamp-2 leading-relaxed">
              {{ model.description }}
            </p>
            <div class="flex items-center gap-4 text-xs text-gray-500 border-t border-white/5 pt-4 mt-auto">
               <span class="flex items-center gap-1.5">
                  <icon-solar-rocket-2-linear class="w-3.5 h-3.5" />
                  {{ gpuName || 'CPU' }}
               </span>
               <span class="flex items-center gap-1.5 ml-auto">
                  <div class="w-1.5 h-1.5 rounded-full" :class="model.public ? 'bg-green-500' : 'bg-red-500'"></div> {{ model.public ? 'Public' : 'Private' }}
               </span>
            </div>
          </div>


          <div class="border border-dashed border-replicate-border p-5 flex flex-col items-center justify-center text-center text-replicate-muted min-h-[180px] hover:bg-replicate-hover/50 hover:border-gray-600 transition-colors cursor-not-allowed group">
             <div class="w-10 h-10 bg-white/5 flex items-center justify-center mb-3 group-hover:bg-white/10 transition-colors">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>
             </div>
             <span class="text-sm font-medium group-hover:text-white transition-colors">Import new model</span>
          </div>
        </div>
      </main>


      <main v-if="currentView === 'gallery'" class="flex-1 overflow-y-auto p-8 animate-fade-in custom-scrollbar">
         <h1 class="text-2xl font-bold text-white mb-6">History & Gallery</h1>

         <div v-if="history.length === 0" class="flex flex-col items-center justify-center py-20 text-gray-500">
            <icon-solar-gallery-linear class="w-16 h-16 opacity-20 mb-4" />
            <p>No history yet. Run a model to see your results here.</p>
         </div>

         <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
             <div v-for="item in history" :key="item.id" class="aspect-square bg-[#111] border border-replicate-border relative group overflow-hidden cursor-pointer" @click="inputImage = item.input; outputImage = item.output; currentView = 'model'">
                 <img :src="item.output" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                 <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center p-4">
                    <span class="text-xs font-mono text-gray-300 mb-2">{{ new Date(item.timestamp).toLocaleTimeString() }}</span>
                    <button class="bg-white text-black px-4 py-2 text-xs font-bold uppercase tracking-wider hover:scale-105 transition-transform">Load Result</button>
                 </div>
             </div>
         </div>
      </main>


      <main v-if="currentView === 'docs'" class="flex-1 overflow-y-auto p-8 animate-fade-in custom-scrollbar max-w-4xl mx-auto">
         <div class="prose prose-invert prose-orange max-w-none">
            <h1>Documentation</h1>
            <p class="lead">How to use the SwinIR Super Resolution Model.</p>

            <h3>Overview</h3>
            <p>SwinIR (Swin Transformer for Image Restoration) is a deep learning model that achieves state-of-the-art performance in image super-resolution. This deployment runs the <strong>Real-SR x4</strong> variant, optimized for upscaling real-world images by a factor of 4.</p>

            <h3>Usage</h3>
            <ul>
               <li><strong>Input</strong>: Upload any JPEG or PNG image. For best results, use images smaller than 1024x1024px to ensure fast processing.</li>
               <li><strong>Process</strong>: The model breaks the image into tiles, processes them in parallel, and stitches them back together.</li>
               <li><strong>Output</strong>: A 4x larger version of your image with restored details and reduced artifacts.</li>
            </ul>

            <h3>API Reference</h3>
            <pre class="bg-[#111] p-4 rounded-none border border-replicate-border text-sm"><code>POST /api/predict
Content-Type: multipart/form-data

file: (binary image data)</code></pre>
         </div>
      </main>


      <main v-if="currentView === 'support'" class="flex-1 overflow-y-auto p-8 animate-fade-in custom-scrollbar">
         <div class="max-w-2xl mx-auto text-center py-20">
            <h1 class="text-3xl font-bold text-white mb-4">Support</h1>
            <p class="text-gray-400 mb-8">Need help with your deployment or have questions about SwinIR?</p>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-left">
               <div class="p-6 border border-replicate-border bg-[#111] hover:bg-[#1a1a1a] transition-colors group cursor-pointer">
                  <h3 class="font-bold text-white mb-2 group-hover:text-orange-500 transition-colors">Documentation</h3>
                  <p class="text-sm text-gray-500">Read the technical integration guides.</p>
               </div>
               <a href="https://github.com/shuretokki/vdronerez-swinir/issues" target="_blank" class="p-6 border border-replicate-border bg-[#111] hover:bg-[#1a1a1a] transition-colors group cursor-pointer block">
                  <h3 class="font-bold text-white mb-2 group-hover:text-blue-500 transition-colors">GitHub Issues</h3>
                  <p class="text-sm text-gray-500">Report bugs or request features.</p>
               </a>
            </div>
         </div>
      </main>


      <main v-if="currentView === 'model'" class="flex-1 flex flex-col overflow-hidden animate-fade-in">


        <div class="px-8 py-6 border-b border-replicate-border bg-[#0f0f0f]">
           <div class="flex items-center gap-2 text-sm text-gray-500 mb-2">
              <span class="hover:text-white cursor-pointer" @click="currentView = 'dashboard'">shuretokki</span>
              <span class="text-gray-700">/</span>
              <span class="text-white font-medium">vdrone-swinir-x4</span>
           </div>
           <div class="flex items-center justify-between">
              <h1 class="text-2xl font-bold text-white tracking-tight">vdrone-swinir-x4</h1>
               <div class="flex gap-2">
                  <button @click="triggerRunOrUpload" class="px-3 py-1.5 text-xs font-medium bg-white text-black rounded hover:bg-gray-200 transition-colors">Run</button>
                  <button @click="showApiModal = true" class="px-3 py-1.5 text-xs font-medium text-gray-400 border border-replicate-border rounded hover:text-white hover:border-gray-600 transition-colors">API</button>
               </div>
           </div>
        </div>


        <div class="flex-1 flex overflow-hidden">


           <div class="w-1/3 border-r border-replicate-border p-8 overflow-y-auto custom-scrollbar bg-[#0f0f0f]">
              <h3 class="text-sm font-bold text-white uppercase tracking-wider mb-6">Input</h3>

              <div class="space-y-6">
                 <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Image</label>
                    <div
                      class="border-2 border-dashed border-replicate-border rounded-lg h-48 flex flex-col items-center justify-center text-center cursor-pointer hover:border-white/30 hover:bg-white/5 transition-all group overflow-hidden relative"
                      @click="fileInput?.click()"
                      @dragover.prevent
                      @drop.prevent="onDrop"
                    >
                       <input ref="fileInput" type="file" class="hidden" accept="image/*" @change="onFileChange" />

                       <img v-if="inputImage" :src="inputImage" class="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-40 transition-opacity" />

                       <div class="relative z-10 flex flex-col items-center">
                          <icon-solar-upload-square-broken v-if="!inputImage" class="w-8 h-8 text-gray-500 mb-2 group-hover:text-white transition-colors" />
                          <span class="text-sm text-gray-400 font-medium group-hover:text-white transition-colors">
                             {{ inputImage ? 'Change Image' : 'Click or drop image' }}
                          </span>
                       </div>
                    </div>
                 </div>


                 <div class="pt-6 border-t border-replicate-border">
                    <button
                       @click="runInference"
                       :disabled="!inputImage || isProcessing"
                       class="w-full py-3 bg-white text-black font-bold rounded hover:bg-gray-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_25px_rgba(255,255,255,0.2)] flex items-center justify-center gap-2"
                    >
                       <svg v-if="isProcessing" class="animate-spin h-4 w-4 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                       </svg>
                       {{ isProcessing ? 'Running...' : 'Run' }}
                    </button>


                    <div v-if="isProcessing" class="mt-4 animate-fade-in">
                       <div class="flex justify-between text-xs text-gray-500 mb-1">
                          <span class="font-mono">{{ progressStage }}</span>
                          <span>{{ progress.toFixed(0) }}%</span>
                       </div>
                       <div class="h-1 bg-gray-800 rounded-full overflow-hidden">
                          <div class="h-full bg-blue-500 transition-all duration-300 ease-out" :style="{ width: progress + '%' }"></div>
                       </div>
                    </div>
                 </div>
              </div>
           </div>


           <div class="w-2/3 bg-[#0b0b0b] p-8 flex flex-col">
              <div class="flex items-center justify-between mb-6">
                 <h3 class="text-sm font-bold text-white uppercase tracking-wider">Output</h3>
                 <div class="flex items-center gap-4">
                    <span v-if="inferenceTime" class="text-xs font-mono text-gray-500">
                       Inference time: <span class="text-white">{{ inferenceTime.toFixed(2) }}s</span>
                    </span>
                    <button
                       v-if="outputImage"
                       @click="downloadResult"
                       class="text-xs font-bold text-white bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded transition-colors"
                    >
                       Download
                    </button>
                 </div>
              </div>

              <div class="flex-1 flex items-center justify-center rounded-xl bg-[#111] border border-replicate-border overflow-hidden relative">


                 <div v-if="!inputImage && !outputImage" class="text-center text-gray-600">
                    <div class="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mx-auto mb-4">
                       <icon-solar-rocket-2-linear class="w-8 h-8 opacity-20" />
                    </div>
                    <p class="font-medium">Run the model to see results</p>
                 </div>


                 <div v-else-if="isProcessing && !outputImage" class="text-center">
                    <div class="w-12 h-12 border-4 border-white/20 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
                    <p class="text-gray-400 text-sm animate-pulse">{{ progressStage }}</p>
                 </div>


                 <div v-else-if="inputImage && outputImage" class="w-full h-full p-4 flex items-center justify-center">
                    <ComparisonSlider :before="inputImage" :after="outputImage" />
                 </div>


                 <div v-else-if="inputImage" class="w-full h-full p-8 flex items-center justify-center">
                    <img :src="inputImage" class="max-w-full max-h-full object-contain opacity-50 grayscale blur-sm transform scale-95" />
                    <div class="absolute inset-0 flex items-center justify-center text-white/50 font-bold tracking-widest uppercase text-xl pointer-events-none">Waiting for Run</div>
                 </div>

              </div>
           </div>

        </div>

      </main>

    </div>

    <!-- API Modal -->
    <div v-if="showApiModal" class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in" @click.self="showApiModal = false">
        <div class="bg-[#1a1a1a] border border-replicate-border rounded-xl w-full max-w-2xl overflow-hidden shadow-2xl transform transition-all duration-300 scale-100 m-4">
            <div class="flex items-center justify-between px-6 py-4 border-b border-replicate-border bg-[#0f0f0f]">
                <h3 class="text-white font-bold">API Usage</h3>
                <button @click="showApiModal = false" class="text-gray-400 hover:text-white transition-colors">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
            </div>
            <div class="p-6 overflow-y-auto max-h-[80vh]">
                <p class="text-gray-400 text-sm mb-4">Run this model via API using curl or your preferred language.</p>
                <div class="mb-4">
                  <span class="text-xs uppercase font-bold text-gray-500 mb-2 block">cURL</span>
                  <pre class="bg-[#0b0b0b] p-4 rounded-lg border border-replicate-border text-xs text-gray-300 font-mono overflow-x-auto whitespace-pre-wrap"><code>curl -X POST http://localhost:8000/api/predict \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/image.jpg" \
      -o output.png</code></pre>
                </div>
            </div>
        </div>
    </div>
  </div>
</template>

<style>

.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: #0f0f0f;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 3px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: #444;
}
</style>
