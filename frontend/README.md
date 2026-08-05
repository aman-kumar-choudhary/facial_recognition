# Checkpoint — Face Authentication Frontend

A minimal Vue 3 UI with two pages, talking to the FastAPI backend:

1. **Register** (`/register`) — capture the student's name, student ID,
   email, and a live photo from the camera; sends it to
   `POST /api/v1/students/register`.
2. **Verify** (`/verify`) — live camera scan; sends a frame to
   `POST /api/v1/auth/authenticate` and shows Authenticated /
   Not Recognized, along with similarity score, liveness score, and latency.

The Verify page is continuous: its camera starts once, sends inexpensive
position checks while idle, and automatically captures for full verification
after the face stays centered. The compact oval is gray with no face, red for
an incorrectly positioned face, and green when it is aligned and ready.

## Setup

```bash
npm install
# edit .env if your backend isn't on localhost:8000
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`). Your
browser will prompt for camera permission on both pages — allow it, since
both registration and verification rely on live capture, not file upload,
matching the "real-time camera authentication" requirement.

## Build for production

```bash
npm run build
```

Outputs static files to `dist/`, deployable to any static host (as long as
it can reach the backend API — set `VITE_API_BASE_URL` accordingly before building).

When deployed with this repository's root `docker-compose.yml`, leave
`VITE_API_BASE_URL` empty. The browser then calls same-origin `/api/...`, and
the edge Nginx proxy forwards those requests to the private backend container.

## Project structure

```
src/
  main.js                 app entrypoint
  App.vue                 top nav shell (Register / Verify tabs)
  api.js                  fetch wrapper for the two backend endpoints
  style.css               design tokens (colors, type, radii)
  components/
    ScanFrame.vue          compact, state-coloured face positioning guide
    CameraCapture.vue       persistent getUserMedia wrapper, shared by both pages
  views/
    RegisterView.vue        page 1: enrollment
    AuthenticateView.vue     page 2: verification
  router/
    index.js                two routes: /register, /verify
```

## Notes

- Camera capture requires a secure context (`https://` or `localhost`) —
  browsers block `getUserMedia` on plain `http://` for any other host. Camera
  access is requested from the device which opens the page, so a phone needs
  to reach the app over HTTPS before it can use that phone's camera.
- CORS: the backend's `app/main.py` currently allows all origins for
  prototype convenience; restrict `allow_origins` before any real deployment.
- Position checks run at a bounded cadence and never overlap. Full liveness
  and recognition requests happen only after two stable green checks, then a
  brief result cooldown keeps the UI responsive without flooding the API.
