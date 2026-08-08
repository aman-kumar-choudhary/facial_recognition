<script setup>
import { ref, computed, onBeforeUnmount, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import CameraCapture from '../components/CameraCapture.vue'
import { assessFacePosition, getStudent, registerStudent, updateStudentFace } from '../api'

const route = useRoute()
const router = useRouter()
const updateStudentId = computed(() => typeof route.query.update === 'string' ? route.query.update : '')
const isUpdate = computed(() => Boolean(updateStudentId.value))

const name = ref('')
const rollNumber = ref('')
const email = ref('')
const capturedImages = ref({})
const cameraRef = ref(null)

const status = ref('idle') // idle | submitting | success | error
const statusMessage = ref('')
const registeredStudent = ref(null)
const cameraReady = ref(false)
const captureState = ref('no-face') // no-face | misaligned | ready | danger
const enrollmentSteps = [
  { key: 'center', label: 'Center face', instruction: 'Look straight at the camera.' },
  { key: 'right', label: 'Right side', instruction: 'Turn your head slightly to your right.' },
  { key: 'left', label: 'Left side', instruction: 'Turn your head slightly to your left.' },
  { key: 'chin_up', label: 'Chin up', instruction: 'Lift your chin slightly.' },
  { key: 'head_down', label: 'Head down', instruction: 'Lower your chin slightly.' },
]
const currentStepIndex = ref(0)
const currentStep = computed(() => enrollmentSteps[currentStepIndex.value])
const guidance = ref(enrollmentSteps[0].instruction)

let positionTimer = null
let positionRequestActive = false
let isActive = true
const POSITION_INTERVAL_MS = 450

const canSubmit = computed(
  () => name.value.trim() && rollNumber.value.trim() && email.value.trim() && currentStepIndex.value === enrollmentSteps.length && status.value !== 'submitting'
)

function handleCapture(dataUrl) {
  if (!currentStep.value) return
  capturedImages.value = { ...capturedImages.value, [currentStep.value.key]: dataUrl }
  currentStepIndex.value += 1
  guidance.value = currentStep.value ? currentStep.value.instruction : 'All five poses captured. Complete registration.'
}

function retake() {
  capturedImages.value = {}
  currentStepIndex.value = 0
  status.value = 'idle'
  statusMessage.value = ''
  captureState.value = 'no-face'
  guidance.value = enrollmentSteps[0].instruction
  schedulePositionCheck(0)
}

function schedulePositionCheck(delay = POSITION_INTERVAL_MS) {
  if (!isActive || !cameraReady.value || !currentStep.value) return
  window.clearTimeout(positionTimer)
  positionTimer = window.setTimeout(checkPosition, delay)
}

async function checkPosition() {
  if (!isActive || !currentStep.value || positionRequestActive || !cameraRef.value) return
  const frame = cameraRef.value.captureFrame({ quality: 0.62, cropToView: false, emitFrame: false })
  if (!frame) {
    schedulePositionCheck()
    return
  }

  positionRequestActive = true
  try {
    const position = await assessFacePosition({ imageBase64: frame, pose: currentStep.value.key })
    captureState.value = position.state === 'ready' ? 'ready' : position.state === 'misaligned' ? 'misaligned' : 'no-face'
    guidance.value = position.state === 'ready' ? currentStep.value.instruction : position.message
  } catch (err) {
    // Keep capture manual: this only updates the guide and never takes a photo.
    captureState.value = 'danger'
    guidance.value = err.message || 'Unable to assess face position.'
  } finally {
    positionRequestActive = false
  }
  schedulePositionCheck()
}

function handleCameraReady() {
  cameraReady.value = true
  schedulePositionCheck(0)
}

function handleCameraError() {
  cameraReady.value = false
  captureState.value = 'danger'
  window.clearTimeout(positionTimer)
}

function capturePhoto() {
  if (!cameraReady.value) return
  cameraRef.value?.captureFrame({ cropToView: false })
}

async function submit() {
  status.value = 'submitting'
  statusMessage.value = ''
  try {
    const result = isUpdate.value
      ? await updateStudentFace(updateStudentId.value, capturedImages.value)
      : await registerStudent({ name: name.value.trim(), rollNumber: rollNumber.value.trim(), email: email.value.trim(), enrollmentImages: capturedImages.value })
    registeredStudent.value = result
    status.value = 'success'
  } catch (err) {
    status.value = 'error'
    statusMessage.value = err.message || 'Registration failed. Try again.'
  }
}

function resetForm() {
  name.value = ''
  rollNumber.value = ''
  email.value = ''
  capturedImages.value = {}
  currentStepIndex.value = 0
  status.value = 'idle'
  statusMessage.value = ''
  registeredStudent.value = null
  if (isUpdate.value) router.push({ name: 'manage' })
}

async function loadStudentForUpdate() {
  if (!isUpdate.value) return
  status.value = 'submitting'
  try {
    const student = await getStudent(updateStudentId.value)
    name.value = student.name
    rollNumber.value = student.roll_number
    email.value = student.email
    status.value = 'idle'
    guidance.value = 'Student details loaded. Capture five replacement poses.'
  } catch (err) {
    status.value = 'error'
    statusMessage.value = err.message || 'Could not load student details.'
  }
}

onMounted(loadStudentForUpdate)

onBeforeUnmount(() => {
  isActive = false
  window.clearTimeout(positionTimer)
})

</script>

<template>
  <div class="page">
    <div class="intro">
      <p class="eyebrow">Step 01 — {{ isUpdate ? 'Face replacement' : 'Enrollment' }}</p>
      <h1>{{ isUpdate ? 'Update student face' : 'Register a student' }}</h1>
      <p class="lede">
        {{ isUpdate ? 'Student identity is locked. Capture five new poses to replace the existing embeddings.' : 'Capture five live poses to improve recognition from different angles.' }}
      </p>
    </div>

    <div class="layout">
      <div class="camera-col">
        <template v-if="currentStep">
          <CameraCapture
            ref="cameraRef"
            :state="captureState"
            @frame-captured="handleCapture"
            @camera-ready="handleCameraReady"
            @camera-error="handleCameraError"
          />
          <p class="guidance" :class="captureState">Pose {{ currentStepIndex + 1 }} of 5 — {{ currentStep.label }}. {{ guidance }}</p>
          <button class="btn primary full" :disabled="!cameraReady || captureState !== 'ready'" @click="capturePhoto">
            Capture {{ currentStep.label }}
          </button>
        </template>
        <template v-else>
          <div class="scan-frame-static">
            <img :src="capturedImages.center" alt="Captured center-face student photo" />
          </div>
          <button class="btn ghost full" @click="retake" :disabled="status === 'submitting'">
            Retake all poses
          </button>
        </template>
      </div>

      <div class="form-col">
        <label class="field">
          <span class="field-label">Full name</span>
          <input v-model="name" type="text" placeholder="Aarav Sharma" :disabled="status === 'submitting' || isUpdate" />
        </label>

        <label class="field">
          <span class="field-label">Roll number (Student ID)</span>
          <input v-model="rollNumber" type="text" placeholder="2026-CS-014" :disabled="status === 'submitting' || isUpdate" />
        </label>

        <label class="field">
          <span class="field-label">Email</span>
          <input v-model="email" type="email" placeholder="aarav.sharma@example.edu" :disabled="status === 'submitting' || isUpdate" />
        </label>

        <button class="btn primary full" :disabled="!canSubmit" @click="submit">
          <span v-if="status === 'submitting'">{{ isUpdate ? 'Updating…' : 'Registering…' }}</span>
          <span v-else>{{ isUpdate ? 'Replace face embeddings' : 'Register student' }}</span>
        </button>

        <p v-if="status === 'error'" class="feedback error">{{ statusMessage }}</p>

        <div v-if="status === 'success'" class="feedback success-card">
          <p class="success-title">{{ isUpdate ? 'Face enrollment updated' : 'Registered' }}</p>
          <dl class="result-grid">
            <template v-if="!isUpdate"><dt>Name</dt><dd>{{ registeredStudent.name }}</dd></template>
            <dt>Student ID / Roll number</dt>
            <dd class="mono">{{ registeredStudent.student_id || updateStudentId }}</dd>
            <template v-if="!isUpdate"><dt>Capture quality</dt><dd class="mono">{{ (registeredStudent.embedding_quality_score * 100).toFixed(1) }}%</dd></template>
            <template v-else><dt>New embedding check</dt><dd class="mono">{{ registeredStudent.similarity_check != null ? `${(registeredStudent.similarity_check * 100).toFixed(1)}%` : '—' }}</dd></template>
          </dl>
          <button class="btn ghost full" @click="resetForm">{{ isUpdate ? 'Back to student management' : 'Register another student' }}</button>
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

.camera-col,
.form-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.scan-frame-static {
  width: min(100%, 320px);
  aspect-ratio: 3 / 4;
  margin: 0 auto;
  border-radius: 50%;
  overflow: hidden;
  border: 2px solid var(--success-dim);
  background: #05070a;
  box-shadow: 0 0 22px rgba(61, 220, 132, 0.18);
}

.scan-frame-static img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.guidance {
  min-height: 1.3em;
  margin: 0;
  text-align: center;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.guidance.ready {
  color: var(--success);
}

.guidance.misaligned,
.guidance.danger {
  color: var(--danger);
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-label {
  font-size: 0.8rem;
  color: var(--text-muted);
}

input {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  color: var(--text-primary);
  font-size: 0.95rem;
  font-family: var(--font-body);
  transition: border-color 0.15s ease;
}

input::placeholder {
  color: var(--text-faint);
}

input:focus {
  border-color: var(--accent);
  outline: none;
}

input:disabled {
  opacity: 0.6;
}

.btn {
  border: none;
  border-radius: var(--radius-sm);
  padding: 12px 20px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.15s ease, background 0.15s ease;
}

.btn.full {
  width: 100%;
}

.btn.primary {
  background: var(--accent);
  color: #04211e;
}

.btn.primary:disabled {
  background: var(--panel-raised);
  color: var(--text-faint);
  cursor: not-allowed;
}

.btn.ghost {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
}

.btn.ghost:hover:not(:disabled) {
  color: var(--text-primary);
  border-color: var(--text-faint);
}

.feedback {
  font-size: 0.85rem;
  margin: 0;
}

.feedback.error {
  color: var(--danger);
}

.success-card {
  border: 1px solid var(--success-dim);
  background: rgba(61, 220, 132, 0.06);
  border-radius: var(--radius-md);
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.success-title {
  font-family: var(--font-display);
  color: var(--success);
  font-weight: 600;
  margin: 0;
}

.result-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 16px;
  margin: 0;
}

.result-grid dt {
  color: var(--text-muted);
  font-size: 0.82rem;
}

.result-grid dd {
  margin: 0;
  font-size: 0.86rem;
}

.mono {
  font-family: var(--font-mono);
}

@media (max-width: 760px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
