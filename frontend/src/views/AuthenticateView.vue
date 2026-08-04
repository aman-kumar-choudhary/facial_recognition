<script setup>
import { ref, onBeforeUnmount } from 'vue'
import CameraCapture from '../components/CameraCapture.vue'
import { assessFacePosition, authenticateFace } from '../api'

const cameraRef = ref(null)
const scanState = ref('no-face') // no-face | misaligned | ready | scanning | success | danger
const result = ref(null)
const errorMessage = ref('')
const guidance = ref('Waiting for a face…')

let pollTimer = null
let positionRequestActive = false
let recognitionRequestActive = false
let consecutiveReadyFrames = 0
let isActive = true
const POSITION_INTERVAL_MS = 450
const RESULT_COOLDOWN_MS = 1800

function schedulePositionCheck(delay = POSITION_INTERVAL_MS) {
  if (!isActive) return
  window.clearTimeout(pollTimer)
  pollTimer = window.setTimeout(checkPosition, delay)
}

async function checkPosition() {
  if (!isActive || positionRequestActive || recognitionRequestActive || !cameraRef.value) return
  const frame = cameraRef.value.captureFrame({ quality: 0.62, cropToView: false, emitFrame: false })
  if (!frame) {
    schedulePositionCheck()
    return
  }

  positionRequestActive = true
  try {
    const position = await assessFacePosition({ imageBase64: frame })
    errorMessage.value = ''
    guidance.value = position.message
    if (position.state === 'ready') {
      scanState.value = 'ready'
      consecutiveReadyFrames += 1
      if (consecutiveReadyFrames >= 2) {
        await recognizeCurrentFace()
        return
      }
    } else {
      scanState.value = position.state === 'misaligned' ? 'misaligned' : 'no-face'
      consecutiveReadyFrames = 0
    }
  } catch (err) {
    // The next frame can recover from transient networking/server errors.
    scanState.value = 'danger'
    guidance.value = err.message || 'Unable to assess face position.'
    consecutiveReadyFrames = 0
  } finally {
    positionRequestActive = false
  }
  schedulePositionCheck()
}

async function recognizeCurrentFace() {
  if (recognitionRequestActive || !cameraRef.value) return
  recognitionRequestActive = true
  scanState.value = 'scanning'
  result.value = null
  errorMessage.value = ''
  try {
    const frame = cameraRef.value.captureFrame({ quality: 0.88, cropToView: false, emitFrame: false })
    if (!frame) return
    const res = await authenticateFace({ imageBase64: frame })
    result.value = res
    errorMessage.value = ''
    scanState.value = res.authenticated ? 'success' : 'danger'
    guidance.value = res.message
  } catch (err) {
    errorMessage.value = err.message || 'Verification request failed.'
    scanState.value = 'danger'
    guidance.value = 'Unable to verify. Reposition and try again.'
  } finally {
    recognitionRequestActive = false
    consecutiveReadyFrames = 0
    schedulePositionCheck(RESULT_COOLDOWN_MS)
  }
}

function handleCameraError(message) {
  errorMessage.value = message
  isActive = false
  window.clearTimeout(pollTimer)
}

onBeforeUnmount(() => {
  isActive = false
  window.clearTimeout(pollTimer)
})
</script>

<template>
  <div class="page">
    <div class="intro">
      <p class="eyebrow">Step 02 — Verification</p>
      <h1>Verify identity</h1>
      <p class="lede">
        Position one face in the oval. Verification starts automatically once
        the guide turns green and keeps monitoring while the camera stays open.
      </p>
    </div>

    <div class="layout">
      <div class="camera-col">
        <CameraCapture ref="cameraRef" :state="scanState" @camera-ready="schedulePositionCheck(0)" @camera-error="handleCameraError" />
        <p class="guidance" :class="scanState">{{ guidance }}</p>
      </div>

      <div class="result-col">
        <div v-if="!result && !errorMessage" class="placeholder" :class="{ scanning: scanState === 'scanning' }">
          <p>{{ scanState === 'scanning' ? 'Running liveness and recognition…' : 'Live results will appear here.' }}</p>
        </div>

        <div v-else-if="result && result.authenticated" class="result-card success">
          <p class="result-title">Authenticated</p>
          <p class="result-name">{{ result.name }}</p>
          <dl class="metric-grid">
            <dt>Similarity</dt>
            <dd class="mono">{{ result.similarity_score?.toFixed(3) }}</dd>
            <dt>Liveness</dt>
            <dd class="mono">{{ result.liveness_score?.toFixed(3) }}</dd>
            <dt>Latency</dt>
            <dd class="mono">{{ result.latency_ms?.toFixed(1) }} ms</dd>
            <dt>Detection</dt>
            <dd class="mono">{{ result.step_latencies_ms?.detection?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Alignment</dt>
            <dd class="mono">{{ result.step_latencies_ms?.alignment?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Liveness</dt>
            <dd class="mono">{{ result.step_latencies_ms?.liveness?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Recognition</dt>
            <dd class="mono">{{ result.step_latencies_ms?.recognition?.toFixed(1) ?? '—' }} ms</dd>
          </dl>
        </div>

        <div v-else-if="result && !result.authenticated" class="result-card danger">
          <p class="result-title">{{ result.message || 'Not recognized' }}</p>
          <dl class="metric-grid" v-if="result.similarity_score != null || result.liveness_score != null || result.face_visibility_score != null">
            <template v-if="result.face_visibility_score != null">
              <dt>Face visibility</dt>
              <dd class="mono">{{ (result.face_visibility_score * 100).toFixed(0) }}%</dd>
            </template>
            <template v-if="result.similarity_score != null">
              <dt>Similarity</dt>
              <dd class="mono">{{ result.similarity_score.toFixed(3) }}</dd>
            </template>
            <template v-if="result.liveness_score != null">
              <dt>Liveness</dt>
              <dd class="mono">{{ result.liveness_score.toFixed(3) }}</dd>
            </template>
            <dt>Latency</dt>
            <dd class="mono">{{ result.latency_ms?.toFixed(1) }} ms</dd>
            <dt>Detection</dt>
            <dd class="mono">{{ result.step_latencies_ms?.detection?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Alignment</dt>
            <dd class="mono">{{ result.step_latencies_ms?.alignment?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Liveness</dt>
            <dd class="mono">{{ result.step_latencies_ms?.liveness?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Recognition</dt>
            <dd class="mono">{{ result.step_latencies_ms?.recognition?.toFixed(1) ?? '—' }} ms</dd>
          </dl>
        </div>

        <div v-if="errorMessage" class="result-card danger">
          <p class="result-title">Request failed</p>
          <p class="result-detail">{{ errorMessage }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  width: 100%;
  max-width: 980px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 36px;
}

.eyebrow {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 10px;
}

h1 {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 600;
  margin: 0 0 10px;
  letter-spacing: -0.01em;
}

.lede {
  color: var(--text-muted);
  max-width: 52ch;
  line-height: 1.55;
  margin: 0;
}

.layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 28px;
  align-items: start;
}

.camera-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.guidance {
  margin: 0;
  min-height: 1.3em;
  text-align: center;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.guidance.ready,
.guidance.scanning,
.guidance.success {
  color: var(--success);
}

.guidance.misaligned,
.guidance.danger {
  color: var(--danger);
}

.result-col {
  min-height: 320px;
  display: flex;
}

.placeholder {
  border: 1px dashed var(--border);
  border-radius: var(--radius-lg);
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 32px;
  color: var(--text-faint);
  font-size: 0.9rem;
  text-align: center;
}

.placeholder.scanning {
  color: var(--accent);
  border-color: var(--accent-dim);
}

.result-card {
  width: 100%;
  border-radius: var(--radius-lg);
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  border: 1px solid var(--border);
}

.result-card.success {
  border-color: var(--success-dim);
  background: rgba(61, 220, 132, 0.06);
}

.result-card.danger {
  border-color: var(--danger-dim);
  background: rgba(240, 97, 107, 0.06);
}

.result-title {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 1.1rem;
  margin: 0;
}

.result-card.success .result-title {
  color: var(--success);
}

.result-card.danger .result-title {
  color: var(--danger);
}

.result-name {
  font-size: 1.6rem;
  font-weight: 600;
  margin: 0;
}

.result-detail {
  color: var(--text-muted);
  font-size: 0.85rem;
  margin: 0;
}

.metric-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 8px 20px;
  margin: 0;
  padding-top: 8px;
  border-top: 1px solid var(--border-soft);
}

.metric-grid dt {
  color: var(--text-muted);
  font-size: 0.82rem;
}

.metric-grid dd {
  margin: 0;
  font-size: 0.9rem;
}

.mono {
  font-family: var(--font-mono);
}

.btn {
  border: none;
  border-radius: var(--radius-sm);
  padding: 12px 20px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
}

.btn.full {
  width: 100%;
}

.btn.primary {
  background: var(--accent);
  color: #04211e;
}

.btn.ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
}

.btn.ghost:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

@media (max-width: 760px) {
  .layout {
    grid-template-columns: 1fr;
  }
  .result-col {
    min-height: 220px;
  }
}
</style>
