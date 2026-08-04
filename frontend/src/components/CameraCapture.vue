<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import ScanFrame from './ScanFrame.vue'

const props = defineProps({
  state: { type: String, default: 'no-face' }, // passed through to ScanFrame
})

const emit = defineEmits(['frame-captured', 'camera-error', 'camera-ready'])

const videoEl = ref(null)
const canvasEl = ref(null)
const stream = ref(null)
const cameraReady = ref(false)
const cameraError = ref('')

async function startCamera() {
  try {
    stream.value = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
    if (videoEl.value) {
      videoEl.value.srcObject = stream.value
      await videoEl.value.play()
      cameraReady.value = true
      console.info(JSON.stringify({ event: 'camera_started' }))
      emit('camera-ready')
    }
  } catch (err) {
    cameraError.value =
      err.name === 'NotAllowedError'
        ? 'Camera access denied. Allow camera permission and reload.'
        : 'Could not access a camera on this device.'
    emit('camera-error', cameraError.value)
  }
}

function stopCamera() {
  stream.value?.getTracks().forEach((track) => track.stop())
  if (stream.value) console.info(JSON.stringify({ event: 'camera_stopped' }))
  stream.value = null
  cameraReady.value = false
}

function captureFrame({ quality = 0.88, cropToView = true, emitFrame = true } = {}) {
  if (!cameraReady.value || !videoEl.value || !canvasEl.value) return
  const video = videoEl.value
  const canvas = canvasEl.value
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight
  if (!sourceWidth || !sourceHeight) return

  // The viewfinder is a portrait oval, but verification can submit the full
  // source frame. Keeping its surrounding context is important to
  // MiniFASNet, especially when a face is close to a frame edge.
  const viewAspectRatio = 3 / 4
  const sourceAspectRatio = sourceWidth / sourceHeight
  let sourceX = 0
  let sourceY = 0
  let cropWidth = sourceWidth
  let cropHeight = sourceHeight

  if (cropToView) {
    if (sourceAspectRatio > viewAspectRatio) {
      cropWidth = sourceHeight * viewAspectRatio
      sourceX = (sourceWidth - cropWidth) / 2
    } else {
      cropHeight = sourceWidth / viewAspectRatio
      sourceY = (sourceHeight - cropHeight) / 2
    }
  }

  canvas.width = Math.round(cropWidth)
  canvas.height = Math.round(cropHeight)
  const ctx = canvas.getContext('2d')
  ctx.drawImage(video, sourceX, sourceY, cropWidth, cropHeight, 0, 0, canvas.width, canvas.height)
  const dataUrl = canvas.toDataURL('image/jpeg', quality)
  if (emitFrame) emit('frame-captured', dataUrl)
  return dataUrl
}

defineExpose({ captureFrame })

onMounted(startCamera)
onBeforeUnmount(stopCamera)
</script>

<template>
  <ScanFrame :state="state">
    <video ref="videoEl" class="feed" playsinline muted></video>
    <canvas ref="canvasEl" class="hidden-canvas"></canvas>
    <p v-if="cameraError" class="camera-error">{{ cameraError }}</p>
    <p v-else-if="!cameraReady" class="camera-loading">Requesting camera…</p>
  </ScanFrame>
</template>

<style scoped>
.feed {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1); /* mirror, like a normal webcam preview */
}

.hidden-canvas {
  display: none;
}

.camera-error,
.camera-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 0 32px;
  font-size: 0.85rem;
  color: var(--text-muted);
  z-index: 3;
}

.camera-error {
  color: var(--danger);
}
</style>
