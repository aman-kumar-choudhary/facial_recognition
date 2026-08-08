<script setup>
import { ref, onBeforeUnmount, onMounted } from 'vue'
import CameraCapture from '../components/CameraCapture.vue'
import { assessFacePosition, authenticateFace, getModelStatus, setActiveModels } from '../api'

const cameraRef = ref(null)
const scanState = ref('no-face') // no-face | misaligned | ready | scanning | success | danger
const result = ref(null)
const attemptLog = ref([])
const errorMessage = ref('')
const guidance = ref('Waiting for a face…')
const modelStatus = ref({ active: { detection: 'SCRFD', liveness: 'MiniFASNet', recognition: 'ArcFace' }, available: {} })
const evaluationRows = ref([])
const modelError = ref('')
const switchingModels = ref(false)

let pollTimer = null
let positionRequestActive = false
let recognitionRequestActive = false
let consecutiveReadyFrames = 0
let isActive = true
const POSITION_INTERVAL_MS = 450
const RESULT_COOLDOWN_MS = 1800

const reasonText = {
  authenticated: 'Authentication successful',
  similarity_below_threshold: 'Similarity below threshold',
  no_matching_enrollment: 'No matching enrolled face',
  spoof_detected: 'Spoof detected',
  face_partially_occluded: 'Face partially occluded',
  matched_student_record_missing: 'Matched student record is unavailable',
}

function resetAttempt(lines = []) {
  result.value = null
  errorMessage.value = ''
  attemptLog.value = lines
}

// function buildAttemptLog(res) {
//   const lines = ['Face detected']
//   if (res.face_visibility_score != null) lines.push(`Face visibility: ${(res.face_visibility_score * 100).toFixed(0)}%`)
//   if (res.liveness_score != null) lines.push(res.reason === 'spoof_detected' ? 'Liveness check failed' : 'Liveness check passed')
//   if (res.similarity_score != null) lines.push(`Similarity: ${(res.similarity_score * 100).toFixed(2)}%`)
//   lines.push(res.authenticated ? 'Authentication successful' : 'Authentication failed')
//   if (!res.authenticated && res.reason) lines.push(`Reason: ${reasonText[res.reason] || res.message}`)
//   return lines
// }

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
      // resetAttempt(['Face detected'])
      scanState.value = 'ready'
      consecutiveReadyFrames += 1
      if (consecutiveReadyFrames >= 2) {
        await recognizeCurrentFace()
        return
      }
    } else {
      scanState.value = position.state === 'misaligned' ? 'misaligned' : 'no-face'
      consecutiveReadyFrames = 0
      if (position.state === 'no_face') {
        resetAttempt(['No face detected', 'Waiting for user…'])
      } else {
        resetAttempt(['Face detected', `Waiting: ${position.message}`])
      }
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
  // resetAttempt(['Face detected', 'Running liveness check…'])
  try {
    const frame = cameraRef.value.captureFrame({ quality: 0.88, cropToView: false, emitFrame: false })
    if (!frame) return
    const res = await authenticateFace({ imageBase64: frame })
    result.value = res
    // attemptLog.value = buildAttemptLog(res)
    evaluationRows.value = [{
      id: `${Date.now()}-${Math.random()}`, timestamp: new Date().toLocaleTimeString(),
      models: res.models || modelStatus.value.active, visibility: res.face_visibility_score,
      similarity: res.similarity_score, liveness: res.liveness_score, timings: res.step_latencies_ms || {},
      total: res.step_latencies_ms?.total ?? res.latency_ms, person: res.name || '—', decision: res.authenticated ? 'Accepted' : 'Rejected',
    }, ...evaluationRows.value].slice(0, 100)
    errorMessage.value = ''
    scanState.value = res.authenticated ? 'success' : 'danger'
    guidance.value = res.message
  } catch (err) {
    errorMessage.value = err.message || 'Verification request failed.'
    attemptLog.value = ['Authentication failed', `Reason: ${errorMessage.value}`]
    scanState.value = 'danger'
    guidance.value = 'Unable to verify. Reposition and try again.'
  } finally {
    recognitionRequestActive = false
    consecutiveReadyFrames = 0
    schedulePositionCheck(RESULT_COOLDOWN_MS)
  }
}

async function changeModel(stage, event) {
  const previous = modelStatus.value.active[stage]
  const selected = event.target.value
  if (selected === previous) return
  switchingModels.value = true
  modelError.value = ''
  try {
    modelStatus.value = await setActiveModels({ [stage]: selected })
  } catch (err) {
    modelError.value = err.message || 'Model change failed.'
    event.target.value = previous
  } finally {
    switchingModels.value = false
  }
}

function optionsFor(stage) { return modelStatus.value.available?.[stage] || [] }
function percent(value, digits = 2) { return value == null ? '—' : `${(value * 100).toFixed(digits)}%` }

onMounted(async () => {
  try { modelStatus.value = await getModelStatus() } catch (err) { modelError.value = err.message || 'Could not load models.' }
})

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

    <section class="model-panel" aria-label="Live model selection">
      <div>
        <p class="panel-label">Active pipeline</p>
        <p class="panel-note">Changes apply to the next processed frame.</p>
      </div>
      <label v-for="stage in ['detection', 'liveness', 'recognition']" :key="stage" class="model-field">
        <span>{{ stage }} model</span>
        <select :value="modelStatus.active[stage]" :disabled="switchingModels" @change="changeModel(stage, $event)">
          <option v-for="option in optionsFor(stage)" :key="option.name" :value="option.name" :disabled="!option.available">
            {{ option.name }}{{ option.available ? '' : ' (not configured)' }}
          </option>
        </select>
      </label>
      <p v-if="modelError" class="model-error">{{ modelError }}</p>
    </section>

    <div class="layout">
      <div class="camera-col">
        <CameraCapture ref="cameraRef" :state="scanState" @camera-ready="schedulePositionCheck(0)" @camera-error="handleCameraError" />
        <p class="guidance" :class="scanState">{{ guidance }}</p>
      </div>

      <div class="result-col">
        <div v-if="!result && !errorMessage && !attemptLog.length" class="placeholder" :class="{ scanning: scanState === 'scanning' }">
          <p>{{ scanState === 'scanning' ? 'Running liveness and recognition…' : 'Live results will appear here.' }}</p>
        </div>

        <div v-else-if="result && result.authenticated" class="result-card success">
          <p class="result-title">Authenticated</p>
          <p class="result-name">{{ result.name }}</p>
          <ol class="attempt-log" aria-label="Authentication attempt log">
            <li v-for="line in attemptLog" :key="line">{{ line }}</li>
          </ol>
          <dl class="metric-grid">
            <dt>Face Visibility</dt>
            <dd class="mono">{{ percent(result.face_visibility_score, 0) }}</dd>
            <dt>Similarity Score</dt>
            <dd class="mono">{{ percent(result.similarity_score) }}</dd>
            <dt>Liveness Score</dt>
            <dd class="mono">{{ percent(result.liveness_score) }}</dd>
            <dt>Detection</dt>
            <dd class="mono">{{ result.step_latencies_ms?.detection?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Alignment</dt>
            <dd class="mono">{{ result.step_latencies_ms?.alignment?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Liveness</dt>
            <dd class="mono">{{ result.step_latencies_ms?.liveness?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Recognition</dt>
            <dd class="mono">{{ result.step_latencies_ms?.recognition?.toFixed(1) ?? '—' }} ms</dd>
            <dt>Total Latency</dt>
            <dd class="mono">{{ result.step_latencies_ms?.total?.toFixed(1) ?? result.latency_ms?.toFixed(1) }} ms</dd>
          </dl>
        </div>

        <div v-else-if="result && !result.authenticated" class="result-card danger">
          <p class="result-title">Authentication failed</p>
          <ol class="attempt-log" aria-label="Authentication attempt log">
            <li v-for="line in attemptLog" :key="line">{{ line }}</li>
          </ol>
          <dl class="metric-grid" v-if="result.similarity_score != null || result.liveness_score != null || result.face_visibility_score != null">
            <template v-if="result.face_visibility_score != null">
              <dt>Face visibility</dt>
              <dd class="mono">{{ (result.face_visibility_score * 100).toFixed(0) }}%</dd>
            </template>
            <template v-if="result.similarity_score != null">
              <dt>Similarity</dt>
              <dd class="mono">{{ percent(result.similarity_score) }}</dd>
            </template>
            <template v-if="result.liveness_score != null">
              <dt>Liveness</dt>
              <dd class="mono">{{ percent(result.liveness_score) }}</dd>
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
          <ol class="attempt-log" aria-label="Authentication attempt log">
            <li v-for="line in attemptLog" :key="line">{{ line }}</li>
          </ol>
        </div>

        <div v-else-if="!result && attemptLog.length" class="result-card status">
          <p class="result-title">Awaiting authentication</p>
          <ol class="attempt-log" aria-label="Current camera state">
            <li v-for="line in attemptLog" :key="line">{{ line }}</li>
          </ol>
        </div>
      </div>
    </div>

    <section class="evaluation-panel">
      <div class="evaluation-heading">
        <div><p class="panel-label">Real-time evaluation</p><p class="panel-note">Latest 100 verification inferences.</p></div>
        <span class="row-count">{{ evaluationRows.length }} records</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Time</th><th>Detection</th><th>Liveness</th><th>Recognition</th><th>Visibility</th><th>Similarity</th><th>Live score</th><th>Detect</th><th>Align</th><th>Live</th><th>Recognize</th><th>Total</th><th>Person</th><th>Decision</th></tr></thead>
          <tbody>
            <tr v-if="!evaluationRows.length"><td colspan="14" class="empty-row">Verification results will appear here.</td></tr>
            <tr v-for="row in evaluationRows" :key="row.id">
              <td>{{ row.timestamp }}</td><td>{{ row.models.detection }}</td><td>{{ row.models.liveness }}</td><td>{{ row.models.recognition }}</td>
              <td>{{ percent(row.visibility, 0) }}</td><td>{{ percent(row.similarity) }}</td><td>{{ percent(row.liveness) }}</td>
              <td>{{ row.timings.detection?.toFixed(1) ?? '—' }}</td><td>{{ row.timings.alignment?.toFixed(1) ?? '—' }}</td><td>{{ row.timings.liveness?.toFixed(1) ?? '—' }}</td><td>{{ row.timings.recognition?.toFixed(1) ?? '—' }}</td><td>{{ row.total?.toFixed(1) ?? '—' }}</td><td>{{ row.person }}</td><td :class="row.decision === 'Accepted' ? 'accepted' : 'rejected'">{{ row.decision }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page {
  width: 100%;
  max-width: 1500px;
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
  max-width: 980px;
}

.model-panel, .evaluation-panel { border: 1px solid var(--border); border-radius: var(--radius-lg); background: var(--panel); padding: 18px; }
.model-panel { display: grid; grid-template-columns: 1.25fr repeat(3, 1fr); gap: 14px; align-items: end; }
.panel-label { margin: 0; font-family: var(--font-display); font-weight: 600; }
.panel-note { margin: 4px 0 0; color: var(--text-muted); font-size: .78rem; }
.model-field { display: flex; flex-direction: column; gap: 6px; color: var(--text-muted); font-size: .74rem; text-transform: capitalize; }
select { border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--panel-raised); color: var(--text-primary); padding: 9px; }
.model-error { grid-column: 1 / -1; margin: 0; color: var(--danger); font-size: .82rem; }
.evaluation-panel { display: flex; flex-direction: column; gap: 14px; }
.evaluation-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.row-count { color: var(--text-faint); font: .75rem var(--font-mono); }
.table-wrap { height: min(62vh, 680px); min-height: 480px; overflow: auto; border: 1px solid var(--border-soft); border-radius: var(--radius-sm); }
table { min-width: 1280px; width: 100%; border-collapse: collapse; font: .72rem var(--font-mono); white-space: nowrap; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--border-soft); text-align: left; }
th { position: sticky; top: 0; background: var(--panel-raised); color: var(--text-muted); font-weight: 500; }
.empty-row { text-align: center; color: var(--text-faint); padding: 28px; }.accepted { color: var(--success); }.rejected { color: var(--danger); }

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

.result-card.status {
  background: rgba(69, 214, 196, 0.04);
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

.attempt-log {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 0;
  padding: 10px 0 0 20px;
  border-top: 1px solid var(--border-soft);
  color: var(--text-muted);
  font-size: 0.84rem;
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
    max-width: none;
  }
  .model-panel { grid-template-columns: 1fr; }
  .table-wrap { height: 55vh; min-height: 360px; }
  .result-col {
    min-height: 220px;
  }
}
</style>
