import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import cache_student, invalidate_student
from app.database import get_db_session
from app.ml.face_detector import NoFaceDetectedError, MultipleFacesDetectedError
from app.models_db import Student
from app.schemas import (StudentRegisterRequest, StudentRegisterResponse, StudentFaceUpdateRequest,
    StudentFaceUpdateResponse, StudentDeleteResponse, StudentListItem, StudentDetailResponse)
from app.utils.image_utils import decode_base64_image, InvalidImageError, save_registration_images, ImageReplacement
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
    recognition_model = next(iter(results.values())).models["recognition"]
    vector_store = get_vector_store(recognition_model)
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
            "recognition_model": recognition_model,
        },
    )

    return StudentRegisterResponse(
        student_id=student.student_id,
        name=student.name,
        roll_number=student.roll_number,
        embedding_quality_score=student.embedding_quality_score,
    )


@router.get("", response_model=list[StudentListItem])
async def list_students(q: str = "", db: AsyncSession = Depends(get_db_session)):
    query = select(Student).order_by(Student.updated_at.desc())
    if q.strip():
        term = f"%{q.strip()}%"
        query = query.where((Student.student_id.ilike(term)) | (Student.name.ilike(term)))
    students = (await db.execute(query)).scalars().all()
    store = get_vector_store()
    image_dir = __import__("pathlib").Path(__import__("app.config", fromlist=["settings"]).settings.STUDENT_IMAGE_DIR)
    return [StudentListItem(student_id=s.student_id, name=s.name, roll_number=s.roll_number,
        embedding_count=len(store.embeddings_for_student(s.student_id)),
        image_url=f"/api/v1/students/{s.student_id}/image" if any(image_dir.glob(f"{s.student_id}_*.jpg")) else None,
        last_updated=s.updated_at.isoformat() if s.updated_at else None) for s in students]


@router.get("/{student_id}/image")
async def student_image(student_id: str):
    from fastapi.responses import FileResponse
    from pathlib import Path
    directory = Path(__import__("app.config", fromlist=["settings"]).settings.STUDENT_IMAGE_DIR)
    candidates = sorted(path for path in directory.glob(f"{student_id}_*.jpg") if not path.name.endswith("_gray.jpg"))
    if not candidates:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student image not found")
    return FileResponse(candidates[0], media_type="image/jpeg")


@router.get("/{student_id}", response_model=StudentDetailResponse)
async def get_student(student_id: str, db: AsyncSession = Depends(get_db_session)):
    student = (await db.execute(select(Student).where(Student.student_id == student_id))).scalar_one_or_none()
    if student is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    return StudentDetailResponse(student_id=student.student_id, name=student.name, roll_number=student.roll_number, email=student.email)


@router.put("/{student_id}/face", response_model=StudentFaceUpdateResponse)
async def update_student_face(student_id: str, payload: StudentFaceUpdateRequest, request: Request, db: AsyncSession = Depends(get_db_session)):
    started = time.perf_counter()
    student = (await db.execute(select(Student).where(Student.student_id == student_id))).scalar_one_or_none()
    if student is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    raw_images = payload.face_images or ({"center": payload.image_base64} if payload.image_base64 else None)
    if not raw_images: raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Supply image_base64 or face_images")
    required_poses = {"center", "right", "left", "chin_up", "head_down"}
    if set(raw_images) != required_poses:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Face replacement requires five new poses: center, right, left, chin_up, head_down")
    try:
        images = {pose: decode_base64_image(raw) for pose, raw in raw_images.items()}
    except InvalidImageError as exc: raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    try:
        # A replacement is a full five-pose enrollment, not an authentication
        # attempt.  The authentication visibility/liveness gate is frontal-only
        # and rejects legitimate left/right/chin poses by design. Keep update
        # validation identical to initial registration so these captures can be
        # safely replaced as one set.
        processed = {pose: await run_in_threadpool(request.app.state.pipeline.process_for_registration, image) for pose, image in images.items()}
    except (NoFaceDetectedError, MultipleFacesDetectedError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    model = next(iter(processed.values())).models["recognition"]
    store = get_vector_store(model)
    old_embeddings = store.embeddings_for_student(student_id)
    replacement = ImageReplacement(student_id, images)
    try:
        replacement.prepare()
        embeddings = {f"{pose}_{variant}": vector for pose, result in processed.items() for variant, vector in result.embeddings.items()}
        store.replace_embeddings(student_id, embeddings)
        student.embedding_quality_score = sum(result.quality_score for result in processed.values()) / len(processed)
        replacement.commit()
        await db.commit()
        replacement.finalize()
    except Exception as exc:
        await db.rollback()
        replacement.rollback()
        if old_embeddings: store.replace_embeddings(student_id, old_embeddings)
        else: store.remove(student_id)
        logger.exception("student_face_update_failed", extra={"event": "student_face_update_failed", "student_id": student_id})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not update student face; previous enrollment was restored") from exc
    similarity = max((score for vector in embeddings.values() for sid, score in store.search(vector, 1) if sid == student_id), default=None)
    return StudentFaceUpdateResponse(success=True, student_id=student_id, embedding_updated=True, image_updated=True,
        similarity_check=round(similarity, 4) if similarity is not None else None, latency_ms=round((time.perf_counter()-started)*1000, 2))


@router.delete("/{student_id}", response_model=StudentDeleteResponse)
async def delete_student(student_id: str, db: AsyncSession = Depends(get_db_session)):
    student = (await db.execute(select(Student).where(Student.student_id == student_id))).scalar_one_or_none()
    if student is None: raise HTTPException(status.HTTP_404_NOT_FOUND, "Student not found")
    store = get_vector_store()
    snapshot = store.embeddings_for_student(student_id)
    removal = ImageReplacement(student_id)
    try:
        removal.prepare(); removal.commit()
        store.remove(student_id)
        await db.delete(student); await db.commit()
        removal.finalize()
    except Exception as exc:
        await db.rollback(); removal.rollback()
        if snapshot: store.replace_embeddings(student_id, snapshot)
        logger.exception("student_delete_failed", extra={"event": "student_delete_failed", "student_id": student_id})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not delete student; removal was rolled back") from exc
    try: await invalidate_student(student_id)
    except Exception: logger.warning("student_cache_invalidate_failed", extra={"event": "student_cache_invalidate_failed", "student_id": student_id})
    logger.info("student_deleted", extra={"event": "student_deleted", "student_id": student_id})
    return StudentDeleteResponse(success=True, student_id=student_id, database_deleted=True, images_deleted=True, embeddings_deleted=True)
