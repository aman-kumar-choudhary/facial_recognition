from fastapi import APIRouter, Request
from app.vector_store import get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    pipeline_ready = getattr(request.app.state, "pipeline", None) is not None
    return {"status": "ok", "models_loaded": pipeline_ready, "faiss": get_vector_store().status()}
