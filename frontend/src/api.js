// An explicitly empty production build uses same-origin Nginx /api. When the
// variable is absent altogether, retain the existing local-development URL.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function postJson(path, body, method = 'POST') {
  const startedAt = performance.now()
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'Request failed'
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  console.debug(JSON.stringify({ event: 'api_request_completed', path, latency_ms: +(performance.now() - startedAt).toFixed(1) }))
  return data
}

export function registerStudent({ name, rollNumber, email, enrollmentImages }) {
  return postJson('/api/v1/students/register', {
    name,
    roll_number: rollNumber,
    email,
    enrollment_images: enrollmentImages,
  })
}

export function authenticateFace({ imageBase64 }) {
  return postJson('/api/v1/auth/authenticate', {
    image_base64: imageBase64,
  })
}

export function assessFacePosition({ imageBase64, pose }) {
  return postJson('/api/v1/auth/position', {
    image_base64: imageBase64,
    pose,
  })
}

export function getModelStatus() {
  return fetch(`${BASE_URL}/api/v1/models`).then(async (res) => {
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || 'Could not load model status')
    return data
  })
}

export function setActiveModels(models) {
  return postJson('/api/v1/models/active', models, 'PUT')
}

export async function getStudents(query = '') {
  const res = await fetch(`${BASE_URL}/api/v1/students${query ? `?q=${encodeURIComponent(query)}` : ''}`)
  if (!res.ok) throw new Error('Could not load students')
  return res.json()
}

export function updateStudentFace(studentId, faceImages) {
  return postJson(`/api/v1/students/${encodeURIComponent(studentId)}/face`, { face_images: faceImages }, 'PUT')
}

export async function getStudent(studentId) {
  const res = await fetch(`${BASE_URL}/api/v1/students/${encodeURIComponent(studentId)}`)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || 'Could not load student')
  return data
}

export async function deleteStudent(studentId) {
  const res = await fetch(`${BASE_URL}/api/v1/students/${encodeURIComponent(studentId)}`, { method: 'DELETE' })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || 'Could not delete student')
  return body
}

export async function getSystemStats() {
  const res = await fetch(`${BASE_URL}/api/v1/monitoring/stats`)
  if (!res.ok) throw new Error('Could not load system statistics')
  return res.json()
}
