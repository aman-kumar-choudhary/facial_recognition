"""
Centralized application configuration.
All configuration is loaded from the .env file.
Application startup will fail if any required environment variable is missing.
"""

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

    # Select which representation is persisted and searched.  ``both`` is
    # the legacy/default behaviour and therefore remains backward compatible.
    IMAGE_STORAGE_MODE: Literal["grayscale", "color", "both"] = Field(default="both")

    # Minimum estimated proportion of facial features that must be visible
    # before liveness and recognition are allowed to run.
    FACE_VISIBILITY_THRESHOLD: float = Field(default=0.80, ge=0.0, le=1.0)

    # --- Passive liveness (Silent-Face / MiniFASNet ONNX) ---
    # LIVENESS_MODEL_PATH is kept for deployments that use one legacy model.
    # Prefer the two-model ensemble below: path and scale positions must match.
    LIVENESS_MODEL_PATH: str = Field(default="")
    LIVENESS_MODEL_PATHS: str = Field(default="")
    LIVENESS_MODEL_SCALES: str = Field(default="")
    LIVENESS_REAL_CLASS_INDEX: int = Field(default=1)
    # MiniFASNet scores are camera/domain dependent; this calibrated default
    # rejects clear spoof scores while avoiding false rejections of live RGB
    # webcam frames.
    LIVENESS_THRESHOLD: float = Field(default="")

    # --- Runtime ---
    MAX_UPLOAD_IMAGE_MB: int = Field(...)
    STUDENT_IMAGE_DIR: str = Field(default="./data/student_image")
    LOG_LEVEL: str = Field(default="INFO")
    AUTH_AUDIT_LOG_PATH: str = Field(default="./data/authentication_audit.csv")

    @field_validator("IMAGE_STORAGE_MODE", mode="before")
    @classmethod
    def normalize_image_storage_mode(cls, value: object) -> str:
        return str(value).strip().lower()


settings = Settings()
