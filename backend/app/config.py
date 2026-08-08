"""
Centralized application configuration.
All configuration is loaded from the .env file.
Application startup will fail if any required environment variable is missing.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- General ---
    APP_NAME: str = Field(...)
    ENV: str = Field(...)

    # --- Database ---
    DATABASE_URL: str = Field(...)

    # --- Redis ---
    REDIS_URL: str = Field(...)
    REDIS_STUDENT_TTL_SECONDS: int = Field(...)
    REDIS_AUTH_RESULT_TTL_SECONDS: int = Field(...)

    # --- Vector Store ---
    VECTOR_STORE_BACKEND: str = Field(...)
    VECTOR_INDEX_PATH: str = Field(...)
    VECTOR_META_PATH: str = Field(...)
    EMBEDDING_DIM: int = Field(...)
    # FAISS_DEVICE=auto uses a GPU clone when the installed FAISS build and
    # CUDA driver support it; CPU remains the safe fallback.
    FAISS_DEVICE: Literal["auto", "cpu", "gpu"] = Field(default="auto")
    FAISS_GPU_DEVICE: int = Field(default=0, ge=0)
    FAISS_CANDIDATE_COUNT: int = Field(default=64, ge=1, le=10000)

    # --- Face Detection ---
    # Deprecated compatibility setting. Detection now comes from
    # FACE_MODEL_PACK so Antelope is not downloaded a second time.
    DETECTION_MODEL_PACK: str = Field(default="")
    DETECTION_CTX_ID: int = Field(...)  # CUDA GPU index when CUDA is selected
    DETECTION_MIN_CONFIDENCE: float = Field(...)

    # --- Inference hardware ---
    # auto: CUDA when the CUDA execution provider is installed and usable,
    # otherwise CPU. cuda: require CUDA. cpu: never use CUDA.
    INFERENCE_DEVICE: str = Field(default="auto")

    # --- Face Recognition ---
    # FaceAnalysis downloads this pack once and exposes both SCRFD + ArcFace.
    # Empty retains compatibility with older env files by using
    # RECOGNITION_MODEL_PACK below.
    FACE_MODEL_PACK: str = Field(default="")
    RECOGNITION_MODEL_PACK: str = Field(default="antelopev2")
    # Tune on captures from the deployed camera. 0.55 supports the enrolled
    # partial-face use case while still requiring a meaningful ArcFace match.
    COSINE_SIMILARITY_THRESHOLD: float = Field(default="")
    INSIGHTFACE_MODEL_ROOT: str = Field(...)
    # Every optional backend stores its downloaded weights below this folder.
    # It is deliberately separate from /app/data, which is user data.
    MODEL_ROOT: str = Field(default="./models")

    # Select which representation is persisted and searched.  ``both`` is
    # the legacy/default behaviour and therefore remains backward compatible.
    IMAGE_STORAGE_MODE: Literal["grayscale", "color", "both"] = Field(default="both")

    # Minimum estimated proportion of facial features that must be visible
    # before liveness and recognition are allowed to run.
    FACE_VISIBILITY_THRESHOLD: float = Field(default="", ge=0.0, le=1.0)
    # A face that reaches the image edge is usually a cropped/partial capture.
    # This is expressed relative to the shortest image side, so it works for
    # portrait and landscape camera streams alike.
    FACE_VISIBILITY_FRAME_MARGIN_RATIO: float = Field(default="", ge=0.0, le=0.20)
    # Reject a detector result whose five landmarks do not form a plausible
    # face.  This is intentionally a broad geometric sanity check, not a
    # skin-colour or dark-pixel heuristic.
    FACE_VISIBILITY_MIN_LANDMARK_SPREAD_RATIO: float = Field(default="", gt=0.0, le=1.0)

    # Per-feature normalized local-evidence floors. Eye/mouth evidence is
    # measured as contrast against adjacent skin; nose evidence is measured
    # as local detail. The legacy environment-variable names are retained so
    # existing deployments do not lose their calibrated values.
    FACE_VISIBILITY_EYE_DARK_PIXEL_RATIO: float = Field(default="", ge=0.0, le=1.0)
    FACE_VISIBILITY_NOSE_DARK_PIXEL_RATIO: float = Field(default="", ge=0.0, le=1.0)
    FACE_VISIBILITY_MOUTH_DARK_PIXEL_RATIO: float = Field(default="", ge=0.0, le=1.0)

    # --- Passive liveness (Silent-Face / MiniFASNet ONNX) ---
    # LIVENESS_MODEL_PATH is kept for deployments that use one legacy model.
    # Prefer the two-model ensemble below: path and scale positions must match.
    LIVENESS_MODEL_PATH: str = Field(default="")
    LIVENESS_MODEL_PATHS: str = Field(default="")
    LIVENESS_MODEL_SCALES: str = Field(default="")
    # The configured MiniFASNet exports use class 1 for a bona-fide face.
    LIVENESS_REAL_CLASS_INDEX: int = Field(default=1, ge=0)
    # A real class must win the fused ensemble and meet this confidence. 0.50
    # is the model's natural three-class decision boundary, not a bypass.
    LIVENESS_THRESHOLD: float = Field(default="", gt=0.0, lt=1.0)

    # Optional plug-in model locations.  The bundled deployment keeps the
    # established SCRFD / MiniFASNet / ArcFace trio as its default.  Supplying
    # one of these paths makes that named model selectable at runtime.
    RETINAFACE_MODEL_PATH: str = Field(default="")
    MTCNN_MODEL_PATH: str = Field(default="")
    CDCN_MODEL_PATH: str = Field(default="")
    FACENET_MODEL_PATH: str = Field(default="")
    DEEPFACE_MODEL_PATH: str = Field(default="")
    VGGFACE2_MODEL_PATH: str = Field(default="")

    # CDCN has no standard, maintained ONNX export.  A deployment that opts
    # into it must supply a calibrated binary [spoof, live] ONNX classifier.
    CDCN_CROP_SCALE: float = Field(default="", gt=1.0)
    CDCN_REAL_CLASS_INDEX: int = Field(default=1, ge=0)
    CDCN_THRESHOLD: float = Field(default="", gt=0.0, lt=1.0)

    # --- Runtime ---
    MAX_UPLOAD_IMAGE_MB: int = Field(...)
    STUDENT_IMAGE_DIR: str = Field(default="./data/student_image")
    LOG_LEVEL: str = Field(default="INFO")
    AUTH_AUDIT_LOG_PATH: str = Field(default="./data/authentication_audit.csv")

    @field_validator("IMAGE_STORAGE_MODE", mode="before")
    @classmethod
    def normalize_image_storage_mode(cls, value: object) -> str:
        return str(value).strip().lower()

    @property
    def model_root_path(self) -> Path:
        return Path(self.MODEL_ROOT).resolve()


settings = Settings()
