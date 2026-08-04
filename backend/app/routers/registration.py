import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import cache_student
from app.database import get_db_session
from app.ml.face_detector import NoFaceDetectedError, MultipleFacesDetectedError
from app.models_db import Student
from app.schemas import StudentRegisterRequest, StudentRegisterResponse
from app.utils.image_utils import decode_base64_image, InvalidImageError, save_registration_images
from app.vector_store import get_vector_store

router = APIRouter(prefix="/api/v1/students", tags=["registration"])
logger = logging.getLogger(__name__)


@router.post("/register", response_model=StudentRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_student(
    payload: StudentRegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    started = time.perf_counter()
    logger.info("registration_started", extra={"event": "registration_started", "roll_number": payload.roll_number})
    # Uniqueness checks up front so we don't waste a model pass on a dupe.
    existing = await db.execute(
        select(Student).where(
            (Student.roll_number == payload.roll_number) | (Student.email == payload.email)
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Student with this roll number or email already exists")

    required_poses = {"center", "right", "left", "chin_up", "head_down"}
    supplied_poses = set(payload.enrollment_images)
    if supplied_poses != required_poses:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Enrollment requires exactly these poses: {', '.join(sorted(required_poses))}")
    images: dict[str, object] = {}
    try:
        images = {pose: decode_base64_image(encoded) for pose, encoded in payload.enrollment_images.items()}
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    pipeline = request.app.state.pipeline
    try:
        results = {pose: await run_in_threadpool(pipeline.process_for_registration, image) for pose, image in images.items()}
    except NoFaceDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except MultipleFacesDetectedError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Registration requires exactly one face in frame, found {exc.count}",
        ) from exc

    student = Student(
        student_id=payload.roll_number,
        name=payload.name,
        roll_number=payload.roll_number,
        email=payload.email,
        embedding_quality_score=sum(result.quality_score for result in results.values()) / len(results),
    )
    image_paths = []
    vector_store = get_vector_store()
    try:
        db.add(student)
        await db.flush()
        image_paths = save_registration_images(student.student_id, images)
        embeddings = {
            f"{pose}_{variant}": embedding
            for pose, result in results.items()
            for variant, embedding in result.embeddings.items()
        }
        vector_store.replace_embeddings(
            student.student_id,
            embeddings,
        )
        await db.commit()
    except Exception:
        await db.rollback()
        vector_store.remove(student.student_id)
        for path in image_paths:
            path.unlink(missing_ok=True)
        logger.exception("registration_failed", extra={"event": "registration_failed", "roll_number": payload.roll_number})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not complete registration")

    try:
        await cache_student(
            student.student_id,
            {"student_id": student.student_id, "name": student.name, "roll_number": student.roll_number},
        )
    except Exception:
        # Redis is an optional performance layer; registration is durable in
        # SQLite and FAISS even when the cache is temporarily unavailable.
        logger.warning("student_cache_write_failed", extra={"event": "student_cache_write_failed", "student_id": student.student_id})

    logger.info(
        "registration_completed",
        extra={
            "event": "registration_completed",
            "student_id": student.student_id,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "embedding_count": len(embeddings),
        },
    )

    return StudentRegisterResponse(
        student_id=student.student_id,
        name=student.name,
        roll_number=student.roll_number,
        embedding_quality_score=student.embedding_quality_score,
    )
