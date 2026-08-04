import time
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import get_cached_student, cache_student
from app.config import settings
from app.database import get_db_session
from app.ml.face_detector import NoFaceDetectedError, MultipleFacesDetectedError
from app.ml.pipeline import FaceVisibilityRejected, LivenessRejected
from app.audit_log import write_auth_audit
from app.models_db import Student
from app.schemas import AuthenticateRequest, AuthenticateResponse, FacePositionResponse
from app.utils.image_utils import decode_base64_image, InvalidImageError
from app.vector_store import get_vector_store

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
logger = logging.getLogger(__name__)


@router.post("/position", response_model=FacePositionResponse)
async def assess_position(payload: AuthenticateRequest, request: Request):
    """Detection-only positioning endpoint for the continuous camera loop."""
    try:
        image_bgr = decode_base64_image(payload.image_base64)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    assessment = await run_in_threadpool(
        request.app.state.pipeline.assess_position, image_bgr, payload.pose or "center"
    )
    logger.debug(
        "face_position_assessed",
        extra={"event": "face_position_assessed", "state": assessment.state, "face_count": assessment.face_count, "pose": payload.pose or "center"},
    )
    return FacePositionResponse(**assessment.__dict__)


@router.post("/authenticate", response_model=AuthenticateResponse)
async def authenticate(
    payload: AuthenticateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
):
    start = time.perf_counter()
    logger.info("authentication_started", extra={"event": "authentication_started"})

    try:
        image_bgr = decode_base64_image(payload.image_base64)
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    pipeline = request.app.state.pipeline

    try:
        embeddings, _face, liveness_score, step_timings = await run_in_threadpool(
            pipeline.process_for_authentication, image_bgr
        )
    except NoFaceDetectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except MultipleFacesDetectedError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Only one face may be processed at a time, found {exc.count}",
        ) from exc
    except FaceVisibilityRejected as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        step_timings = exc.timings or {}
        write_auth_audit(authenticated=False, student_id=None, reason="face_partially_occluded", similarity_score=None, liveness_score=None, timings=step_timings, total_ms=elapsed_ms)
        logger.info(
            "recognition_result",
            extra={"event": "recognition_result", "authenticated": False, "reason": "face_partially_occluded", "face_visibility_score": exc.score, "face_visibility_threshold": exc.threshold, "latency_ms": round(elapsed_ms, 2), "step_latencies_ms": step_timings},
        )
        return AuthenticateResponse(
            authenticated=False,
            face_visibility_score=exc.score,
            latency_ms=round(elapsed_ms, 2),
            step_latencies_ms=step_timings,
            message="Face is partially occluded. Please show your complete face.",
        )
    except LivenessRejected as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        step_timings = exc.timings or {}
        write_auth_audit(authenticated=False, student_id=None, reason="liveness_rejected", similarity_score=None, liveness_score=exc.score, timings=step_timings, total_ms=elapsed_ms)
        logger.info(
            "recognition_result",
            extra={
                "event": "recognition_result", "authenticated": False, "reason": "liveness_rejected",
                "liveness_score": round(exc.score, 4), "latency_ms": round(elapsed_ms, 2), "step_latencies_ms": step_timings,
            },
        )
        return AuthenticateResponse(
            authenticated=False,
            liveness_score=exc.score,
            latency_ms=round(elapsed_ms, 2),
            step_latencies_ms=step_timings,
            message="Spoof detected -- liveness check failed",
        )

    vector_store = get_vector_store()
    recognition_started = time.perf_counter()
    # Search each capture representation against both enrolled variants and
    # retain the best score for each student. This supports color, grayscale,
    # and poorly lit cameras without another detector or liveness pass.
    match_scores: dict[str, float] = {}
    for embedding in embeddings.values():
        for student_id, score in vector_store.search(embedding, top_k=3):
            match_scores[student_id] = max(match_scores.get(student_id, -1.0), score)
    matches = sorted(match_scores.items(), key=lambda item: item[1], reverse=True)
    step_timings["recognition"] = round(step_timings.get("recognition", 0) + (time.perf_counter() - recognition_started) * 1000, 2)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if not matches:
        write_auth_audit(authenticated=False, student_id=None, reason="no_vectors", similarity_score=None, liveness_score=liveness_score, timings=step_timings, total_ms=elapsed_ms)
        logger.info(
            "recognition_result",
            extra={"event": "recognition_result", "authenticated": False, "reason": "no_vectors", "latency_ms": round(elapsed_ms, 2), "step_latencies_ms": step_timings},
        )
        return AuthenticateResponse(
            authenticated=False,
            liveness_score=liveness_score,
            latency_ms=round(elapsed_ms, 2),
            step_latencies_ms=step_timings,
            message="User Not Recognized",
        )

    student_id, similarity = matches[0]
    runner_up_similarity = matches[1][1] if len(matches) > 1 else None
    if similarity < settings.COSINE_SIMILARITY_THRESHOLD:
        write_auth_audit(authenticated=False, student_id=None, reason="below_threshold", similarity_score=similarity, liveness_score=liveness_score, timings=step_timings, total_ms=elapsed_ms)
        logger.info(
            "recognition_result",
            extra={
                "event": "recognition_result", "authenticated": False, "reason": "below_threshold",
                "similarity_score": round(similarity, 4), "runner_up_similarity": round(runner_up_similarity, 4) if runner_up_similarity is not None else None, "latency_ms": round(elapsed_ms, 2), "step_latencies_ms": step_timings,
            },
        )
        return AuthenticateResponse(
            authenticated=False,
            similarity_score=round(similarity, 4),
            liveness_score=liveness_score,
            latency_ms=round(elapsed_ms, 2),
            step_latencies_ms=step_timings,
            message="User Not Recognized",
        )

    # Redis-first lookup for the matched student's display info, falling
    # back to the primary DB (and repopulating the cache) on a miss.
    try:
        cached = await get_cached_student(student_id)
    except Exception:
        cached = None
        logger.warning("student_cache_read_failed", extra={"event": "student_cache_read_failed", "student_id": student_id})
    if cached:
        name = cached["name"]
    else:
        result = await db.execute(select(Student).where(Student.student_id == student_id))
        student = result.scalar_one_or_none()
        if student is None:
            write_auth_audit(authenticated=False, student_id=student_id, reason="missing_student", similarity_score=similarity, liveness_score=liveness_score, timings=step_timings, total_ms=elapsed_ms)
            return AuthenticateResponse(
                authenticated=False,
                similarity_score=round(similarity, 4),
                liveness_score=liveness_score,
                latency_ms=round(elapsed_ms, 2),
                step_latencies_ms=step_timings,
                message="User Not Recognized",
            )
        name = student.name
        try:
            await cache_student(
                student_id, {"student_id": student_id, "name": student.name, "roll_number": student.roll_number}
            )
        except Exception:
            logger.warning("student_cache_write_failed", extra={"event": "student_cache_write_failed", "student_id": student_id})

    logger.info(
        "recognition_result",
        extra={
            "event": "recognition_result", "authenticated": True, "student_id": student_id,
            "similarity_score": round(similarity, 4), "runner_up_similarity": round(runner_up_similarity, 4) if runner_up_similarity is not None else None, "liveness_score": round(liveness_score, 4), "latency_ms": round(elapsed_ms, 2), "step_latencies_ms": step_timings,
        },
    )
    write_auth_audit(authenticated=True, student_id=student_id, reason="authenticated", similarity_score=similarity, liveness_score=liveness_score, timings=step_timings, total_ms=elapsed_ms)

    return AuthenticateResponse(
        authenticated=True,
        student_id=student_id,
        name=name,
        similarity_score=round(similarity, 4),
        liveness_score=liveness_score,
        latency_ms=round(elapsed_ms, 2),
        step_latencies_ms=step_timings,
        message="Authenticated",
    )
