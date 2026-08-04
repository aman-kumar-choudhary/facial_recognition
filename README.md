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

Open `http://localhost` (or set `APP_PORT` before starting). All API calls go
through Nginx at `/api/`; `/health` is also proxied for readiness checks.

`face_data` is a Docker-managed persistent volume mounted at `/app/data` in
the backend. It holds SQLite, FAISS, enrollment images, and the audit log, so
rebuilding or recreating the backend does not discard them. Inspect or back it
up with `docker volume inspect facial_recognition_face_data`; do not run
`docker compose down -v` unless you deliberately intend to erase persistent
application and Redis data.

For camera access outside localhost, terminate TLS in front of Nginx (or add
a certificate-aware Nginx configuration): browsers require HTTPS for camera
access on non-localhost origins.
