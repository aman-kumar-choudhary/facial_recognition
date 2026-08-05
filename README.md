# Face Recognition Authentication

## Production deployment

The root Compose file runs the frontend, backend, Redis, and an edge Nginx
proxy. Nginx is the only published service; the frontend and backend have no
host ports. SQLite and FAISS run as part of the backend container (they are
embedded libraries, not independent services) and store their persistent
files in the backend data volume.

```bash
# Set production values in the existing backend/.env, including
# IMAGE_STORAGE_MODE and FACE_VISIBILITY_THRESHOLD.
docker compose up --build -d
```

Open `http://localhost` (or set `APP_PORT` before starting). The public Nginx
and the private frontend Nginx both listen on port 80; only the public Nginx
has a host port. All API calls go through Nginx at `/api/`; `/health` is also
proxied for readiness checks.

### Camera use from another device

Camera access belongs to the device and browser which opened the page. For
example, opening the app on a phone requests that phone's camera, not the
Docker host's webcam. Browsers permit camera access on `localhost`, but block
it on plain HTTP LAN addresses such as `http://192.168.x.x`. To use a phone or
another computer, publish this Nginx endpoint through **HTTPS** with a
certificate trusted by that device, then open `https://your-hostname`.

### GPU inference

The Compose backend requires an NVIDIA GPU and is configured to fail startup
rather than run inference on CPU. Install and configure the NVIDIA Container
Toolkit before starting the stack. Verify the active execution provider with
`docker compose exec backend python -c "import onnxruntime as ort;
print(ort.get_available_providers())"`; the output must include
`CUDAExecutionProvider`.

`face_data` is a Docker-managed persistent volume mounted at `/app/data` in
the backend. It holds SQLite, FAISS, enrollment images, and the audit log, so
rebuilding or recreating the backend does not discard them. Inspect or back it
up with `docker volume inspect facial_recognition_face_data`; do not run
`docker compose down -v` unless you deliberately intend to erase persistent
application and Redis data.

For camera access outside localhost, terminate TLS in front of Nginx (or add
a certificate-aware Nginx configuration): browsers require HTTPS for camera
access on non-localhost origins.
