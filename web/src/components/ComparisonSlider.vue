<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  before: string
  after: string
}>()

const container = ref<HTMLElement | null>(null)
const position = ref(50)
const isDragging = ref(false)

const setPosition = (clientX: number) => {
  if (!container.value) return
  const rect = container.value.getBoundingClientRect()
  const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
  position.value = (x / rect.width) * 100
}

const onMouseDown = (e: MouseEvent) => {
  e.preventDefault()
  isDragging.value = true
  setPosition(e.clientX)
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
}

const onMouseMove = (e: MouseEvent) => {
  if (!isDragging.value) return
  e.preventDefault()
  setPosition(e.clientX)
}

const onMouseUp = () => {
  isDragging.value = false
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', onMouseUp)
}

const onTouchStart = (e: TouchEvent) => {
  e.preventDefault()
  const touch = e.touches[0]
  if (touch) {
    isDragging.value = true
    setPosition(touch.clientX)
    window.addEventListener('touchmove', onTouchMove, { passive: false })
    window.addEventListener('touchend', onTouchEnd)
  }
}

const onTouchMove = (e: TouchEvent) => {
  if (!isDragging.value) return
  e.preventDefault()
  const touch = e.touches[0]
  if (touch) {
    setPosition(touch.clientX)
  }
}

const onTouchEnd = () => {
  isDragging.value = false
  window.removeEventListener('touchmove', onTouchMove)
  window.removeEventListener('touchend', onTouchEnd)
}

onUnmounted(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('mousemove', onMouseMove)
    window.removeEventListener('mouseup', onMouseUp)
    window.removeEventListener('touchmove', onTouchMove)
    window.removeEventListener('touchend', onTouchEnd)
  }
})
</script>

<template>
  <div
    ref="container"
    class="relative w-full aspect-video select-none overflow-hidden cursor-ew-resize group rounded-lg border border-replicate-border bg-[#111]"
    @mousedown="onMouseDown"
    @touchstart="onTouchStart"
  >

    <img
      :src="after"
      class="absolute inset-0 w-full h-full object-contain pointer-events-none select-none"
      draggable="false"
    />


    <div
      class="absolute inset-0 overflow-hidden pointer-events-none select-none"
      :style="{ clipPath: `inset(0 ${100 - position}% 0 0)` }"
    >
      <img
        :src="before"
        class="w-full h-full object-contain pointer-events-none select-none"
        style="image-rendering: pixelated"
        draggable="false"
      />
    </div>


    <div
      class="absolute top-0 bottom-0 w-[1px] bg-white/50 backdrop-blur-sm pointer-events-none z-20"
      :style="{ left: position + '%' }"
    >
      <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8
                  bg-white rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.5)]
                  flex items-center justify-center transform transition-transform
                  group-hover:scale-110 group-active:scale-95 text-black">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 8L22 12L18 16"/>
            <path d="M6 8L2 12L6 16"/>
        </svg>
      </div>
    </div>


    <div
        class="absolute bottom-4 left-4 pointer-events-none transition-opacity duration-200"
        :class="{ 'opacity-0': position < 15 }"
    >
        <div class="bg-black/60 backdrop-blur-md text-white text-[10px] font-bold px-2 py-1 rounded border border-white/10 uppercase tracking-wider">Original</div>
    </div>
    <div
        class="absolute bottom-4 right-4 pointer-events-none transition-opacity duration-200"
        :class="{ 'opacity-0': position > 85 }"
    >
        <div class="bg-white/90 backdrop-blur-md text-black text-[10px] font-bold px-2 py-1 rounded border border-white/10 uppercase tracking-wider shadow-lg">Enhanced</div>
    </div>

  </div>
</template>

<style scoped>
img {
  -webkit-user-drag: none;
  user-select: none;
}
</style>
