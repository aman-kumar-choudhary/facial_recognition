from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class StudentRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    roll_number: str = Field(..., min_length=1, max_length=64)
    email: EmailStr
    # Five live captures, one for each requested enrollment pose.
    enrollment_images: dict[str, str] = Field(..., description="Base64 images keyed by center, right, left, chin_up, head_down")


class StudentRegisterResponse(BaseModel):
    student_id: str
    name: str
    roll_number: str
    embedding_quality_score: float
    message: str = "Student registered successfully"


class AuthenticateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG/PNG frame from camera")
    # Used only by the lightweight positioning endpoint during enrollment.
    pose: Optional[str] = Field(default=None, max_length=32)


class AuthenticateResponse(BaseModel):
    authenticated: bool
    student_id: Optional[str] = None
    name: Optional[str] = None
    similarity_score: Optional[float] = None
    liveness_score: Optional[float] = None
    face_visibility_score: Optional[float] = None
    models: dict[str, str] = Field(default_factory=dict)
    latency_ms: float
    step_latencies_ms: dict[str, float] = Field(default_factory=dict)
    message: str


class FacePositionResponse(BaseModel):
    state: str
    message: str
    face_count: int


class ModelSelectionRequest(BaseModel):
    detection: Optional[str] = None
    liveness: Optional[str] = None
    recognition: Optional[str] = None


class ModelStatusResponse(BaseModel):
    active: dict[str, str]
    available: dict[str, list[dict[str, object]]]


class ErrorResponse(BaseModel):
    detail: str
