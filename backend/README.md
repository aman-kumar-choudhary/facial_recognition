# Face Recognition Authentication Prototype — Backend

A production-oriented FastAPI backend for student face-recognition
authentication, built to the architecture below:

```
Registration:  Camera → Antelope SCRFD (detect) → Align → ArcFace (embed) → store
                                                         ├── SQLite (metadata)
                                                         ├── data/student_image/ (configured image variant)
                                                         └── FAISS (configured embedding variant)

Authentication: Camera → SCRFD (detect, single-face) → visibility gate → Silent-Face MiniFASNet ensemble (liveness)
                       → ArcFace (embed) → FAISS search → cosine similarity
                       → Redis (cache student info) → Authenticated / Not Recognized
```

Frontend is intentionally out of scope here (a minimal Vue.js UI is the
companion piece) — this repo is the backend, which is where the actual
requirements (speed, accuracy, scalability) live.

---

## 1. Stack

| Concern              | Choice                                   |
|-----------------------|-------------------------------------------|
| API framework         | FastAPI (async)                          |
| Face detection + recognition | One shared InsightFace Antelope (`antelopev2`) model pack |
| Liveness               | Silent-Face MiniFASNetV2 + MiniFASNetV1SE ONNX ensemble (fail closed) |
| Embedding store         | FAISS (`IndexFlatIP`, cosine similarity) |
| Primary metadata store  | SQLite via async SQLAlchemy (swap to Postgres by changing `DATABASE_URL`) |
| Cache                   | Redis (student info + auth-result caching) |

---

## 2. Project layout

```
app/
  main.py              FastAPI app + model loading/request logging at startup
  logging_config.py    JSON structured logging
  config.py            All settings, env-overridable
  database.py           Async SQLAlchemy engine/session
  models_db.py           Student ORM model (metadata only, no embeddings)
  schemas.py            Pydantic request/response models
  cache.py              Redis wrapper (student cache, auth-result cache)
  vector_store.py        FAISS wrapper behind a swappable VectorStore interface
  ml/
    model_registry.py    one Antelope download/initialization per API process
    face_detector.py     shared SCRFD wrapper + single-face enforcement
    face_align.py         5-point landmark alignment to 112x112 ArcFace template
    face_recognizer.py     ArcFace embedding + cosine similarity helper
    liveness.py            MiniFASNet wrapper + heuristic fallback
    pipeline.py            Orchestrates detect → align → liveness → embed
  routers/
    registration.py       POST /api/v1/students/register
    authentication.py     POST /api/v1/auth/authenticate
    health.py              GET /health
scripts/
  example_client.py       Minimal script to call both endpoints
requirements.txt
.env
```

---

## 3. Setup

### 3.1 Prerequisites
- Python 3.10–3.12 (the pinned ONNX Runtime build does not support Python 3.14 yet)
- Redis (via Docker, or a local install)
- ~2GB free disk for model weights (auto-downloaded on first run)

### 3.2 Install

```bash
conda create -n face python=3.11
conda activate face
pip install -r requirements.txt
```

### 3.3 Install passive-liveness models

```bash
bash scripts/download_liveness_models.sh
```

The two small ONNX weights are installed under `models/liveness/` and are
ignored by Git. Their download is hash-checked by the script. The supplied
`.env` config uses their required crop scales (`2.7,4.0`) and blocks startup
if either model is absent; it never falls back to an unsafe image-texture
heuristic.

### GPU inference (optional, recommended when CUDA is available)

The backend automatically uses one CUDA execution provider for SCRFD,
MiniFASNet, and ArcFace when `INFERENCE_DEVICE=auto` and CUDA is available.
To require GPU and prevent CPU fallback, set `INFERENCE_DEVICE=cuda` in `.env`.

Run the following inside the same environment used for `uvicorn`:

```bash
bash scripts/enable_gpu_inference.sh
nvidia-smi
```

ONNX Runtime 1.19 GPU wheels require an NVIDIA driver plus CUDA 12.x and
cuDNN 9.x. The setup script installs the matching Python CUDA/cuDNN runtime
packages. If your system instead manages CUDA libraries itself, install a
CUDA 12.x runtime and cuDNN 9.x compatible with the active driver. If CUDA is
unavailable, use `INFERENCE_DEVICE=auto` to keep the API working on CPU while
the host is fixed.

### 3.4 Start Redis (local development only)

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

For the full production stack, use the repository-root Compose file instead:

```bash
cd ..
docker compose up --build -d
```

It starts frontend, backend, Redis, and Nginx. SQLite and FAISS run inside the
backend container and persist in its `face_data` volume; they are libraries
and data files, not standalone server processes.

### 3.5 Configure environment

```bash
# Edit the existing .env file for this backend.
```

Storage and recognition security are configuration-only settings:

```dotenv
# grayscale, color, or both (the backward-compatible default)
IMAGE_STORAGE_MODE=both
# 0.0-1.0; authentication stops before liveness/vector search below this value
FACE_VISIBILITY_THRESHOLD=0.80
# The visibility score combines eye/nose/mouth evidence at the five SCRFD
# landmarks with frame integrity. Authentication stops when the score is below
# FACE_VISIBILITY_THRESHOLD. The component scores are emitted in application
# logs as `face_visibility_assessed` for camera-specific calibration.
FACE_VISIBILITY_FRAME_MARGIN_RATIO=0.025
FACE_VISIBILITY_MIN_LANDMARK_SPREAD_RATIO=0.18
FACE_VISIBILITY_EYE_DARK_PIXEL_RATIO=0.30
FACE_VISIBILITY_NOSE_DARK_PIXEL_RATIO=0.20
FACE_VISIBILITY_MOUTH_DARK_PIXEL_RATIO=0.40

# MiniFASNet V1SE/V2 use bona-fide class 1. Both models are fused before one
# three-class decision; the threshold is applied to the fused real-class score.
LIVENESS_REAL_CLASS_INDEX=1
LIVENESS_THRESHOLD=0.50
```

Switching `IMAGE_STORAGE_MODE` affects new enrollment images and embeddings
and the representation used for subsequent authentication. Existing enrolled
data is intentionally not deleted when the setting changes; re-enroll users
if their stored gallery must be converted to the new single-variant mode.

### 3.6 Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first startup, InsightFace downloads the single `antelopev2` pack to
`./models/` (requires internet once). Its prepared SCRFD and ArcFace objects
are shared by detection, registration, positioning, and authentication; no
second Antelope download or model initialization occurs.

Check it's alive:

```bash
curl http://localhost:8000/health
```

### 3.7 Optional model benchmark backends

The default is unchanged: **SCRFD + MiniFASNet + ArcFace**. To download the
additional supported detector and recognition weights into `models/`, run:

```bash
bash scripts/download_optional_models.sh
```

It installs/preloads MTCNN, RetinaFace, FaceNet (CASIA-WebFace), FaceNet
trained on VGGFace2, and DeepFace. The API exposes their state through
`GET /api/v1/models`; select a fully available combination with:

```bash
curl -X PUT http://localhost:8000/api/v1/models/active \
  -H 'Content-Type: application/json' \
  -d '{"detection":"MTCNN","liveness":"MiniFASNet","recognition":"FaceNet"}'
```

Model selections are atomic and lazy-loaded; if a dependency or weight is
missing, the API returns `409` and leaves the active pipeline untouched.
Each recognition backend has its own FAISS index, so enroll benchmark users
again after changing the recognition model. Do not compare raw cosine scores
or reuse the ArcFace threshold across models without calibration.

`CDCN` is also selectable, but it deliberately needs an operator-supplied,
calibrated binary ONNX classifier at `CDCN_MODEL_PATH`. The official CDCN
release contains research checkpoints rather than a standard safe inference
artifact. Its exact input/output contract is in `models/cdcn/README.md`.

---

## 4. API

### `POST /api/v1/students/register`

```json
{
  "name": "",
  "roll_number": "",
  "email": "",
  "image_base64": ""
}
```

Rejects with `422` if zero or more than one face is detected in the image or
if passive liveness detects a spoof (registration requires exactly one clean,
live face). Returns the generated
`student_id` and the detector's confidence as a rough embedding quality
score.

### `POST /api/v1/auth/authenticate`

```json
{ "image_base64": "<base64 JPEG/PNG frame>" }
```

Response:

```json
{
  "authenticated": true,
  "student_id": "…",
  "name": "",
  "similarity_score": ,
  "liveness_score": ,
  "latency_ms": ,
  "message": "Authenticated"
}
```

If liveness fails: `message: "Spoof detected -- liveness check failed"`.
If the face is occluded below `FACE_VISIBILITY_THRESHOLD`: `message: "Face is
partially occluded. Please show your complete face."` and the response
includes `face_visibility_score`.
If no embedding clears the cosine threshold: `message: "User Not Recognized"`.
If more than one face is in frame: `422` (only one face is processed at a time, per spec).

### `POST /api/v1/auth/position`

Runs only the shared SCRFD detector and returns `no_face`, `misaligned`, or
`ready`. The browser uses it to colour the small face oval and only calls the
full liveness/recognition endpoint after two consecutive `ready` responses.
It intentionally avoids ArcFace and liveness inference for responsive camera
guidance.

### `GET /health`

Returns whether the model pipeline finished loading — useful for readiness probes.

---

## 5. Try it

```bash
python scripts/example_client.py register path/to/face.jpg
python scripts/example_client.py authenticate path/to/face.jpg
```

---

## 6. Fine-tuning on an Indian face dataset

The recognition module (`app/ml/face_recognizer.py`) currently loads a
pretrained `glintr100` (ResNet100/ArcFace) checkpoint via `insightface`.
To swap in a fine-tuned model:

1. Fine-tune ArcFace (ResNet100 backbone) on your Indian face dataset using
   your training pipeline of choice (e.g. `insightface`'s own training repo,
   or a PyTorch ArcFace implementation).
2. Export the fine-tuned weights to ONNX with a `(1, 3, 112, 112) → (1, 512)` signature.
3. Replace the model-loading call in `FaceRecognizer.__init__` with a direct
   `onnxruntime.InferenceSession(path_to_your.onnx)` load, keeping
   `get_embedding()`'s interface (input: 112x112 aligned BGR crop, output:
   L2-normalized 512-D vector) unchanged so nothing else in the pipeline needs to change.
4. Re-tune `COSINE_SIMILARITY_THRESHOLD` in `.env` against a held-out
   validation set (ROC/EER analysis) — fine-tuning shifts the score distribution.

---

## 7. Passive liveness (Silent-Face) — deployment requirements

The API uses the SCRFD detection box for liveness, expands it using the crop
convention required by each Silent-Face model, then performs ArcFace alignment
and embedding **only after** a live decision. MiniFASNet consumes an 80×80
NCHW BGR tensor in the original 0–255 range. The V1SE and V2 logits are
averaged before one softmax/three-class decision; one model's lower raw score
is not incorrectly treated as a veto. Liveness is deliberately applied only
during authentication, so a camera-captured enrollment image is not blocked
by the verification-only anti-spoof check.

The model ensemble catches common 2D presentation attacks such as printed
photos and screen replays, but an RGB-only passive model is not a guarantee
against every high-quality mask, deepfake, or camera/device domain shift. The
upstream project likewise requires camera-captured, well-lit, mostly frontal
faces. Before a campus deployment, tune `LIVENESS_THRESHOLD` using real
captures from the target cameras and both bona-fide students and presentation
attacks. The shipped `0.50` is the model's default decision boundary, not a
certified operating point.

---

## 8. Design notes / why things are structured this way

- **Embeddings live in FAISS, not the primary DB.** `students.db` only
  stores identity metadata (name, roll number, email). This keeps the
  metadata store lightweight and lets the vector index be swapped
  (Milvus/Qdrant/Chroma) independently — see `VectorStore` interface in
  `app/vector_store.py`.
- **Redis is a cache in front of the metadata DB**, not the similarity
  search engine. On a match, student display info is served from Redis
  when warm, falling back to SQLite and repopulating the cache on a miss.
- **Models load once at process startup** (FastAPI `lifespan`), not
  per-request. A single Antelope registry owns both SCRFD and ArcFace,
  eliminating duplicated download/initialization work.
- **Enrollment references follow `IMAGE_STORAGE_MODE`.** `both` saves color
  and grayscale image/vector variants; `color` and `grayscale` save only the
  selected representation. Authentication creates only the corresponding
  query embedding(s).
- **Logs are JSON.** Model lifecycle, API timing, liveness/embedding work,
  image storage, vector operations, recognition decisions, and errors carry
  an `event` field. Set `LOG_LEVEL=DEBUG` for frame-level diagnosis.
- **Single-face enforcement** happens at the detector level
  (`detect_single`) and raises a distinct `MultipleFacesDetectedError`,
  making the "only one face processed at a time" requirement an explicit,
  testable code path rather than an implicit assumption.

---

## 9. Scaling beyond the prototype

- Swap SQLite → Postgres by changing `DATABASE_URL` (async driver already SQLAlchemy-abstracted).
- Swap FAISS → Milvus/Qdrant for horizontally-scalable, persistent, multi-node vector search — implement one more `VectorStore` subclass.
- Put the model pipeline behind a dedicated inference service (e.g. Triton/BentoML) if you need to scale inference independently from the API layer.
- Add rate limiting / auth-result caching (`cache_auth_result` is stubbed in `app/cache.py`) to suppress duplicate frame submissions from a live camera loop.
