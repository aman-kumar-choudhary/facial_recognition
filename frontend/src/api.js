// An explicitly empty production build uses same-origin Nginx /api. When the
// variable is absent altogether, retain the existing local-development URL.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function postJson(path, body) {
  const startedAt = performance.now()
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
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
