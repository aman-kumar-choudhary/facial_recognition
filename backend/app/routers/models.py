"""HTTP surface for live model selection."""
from fastapi import APIRouter, HTTPException, Request

from app.ml.model_manager import ModelUnavailableError
from app.schemas import ModelSelectionRequest, ModelStatusResponse

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ModelStatusResponse)
async def get_models(request: Request):
    return request.app.state.model_manager.status()


@router.put("/active", response_model=ModelStatusResponse)
async def set_active_models(payload: ModelSelectionRequest, request: Request):
    try:
        return request.app.state.model_manager.select(**payload.model_dump())
    except (ValueError, ModelUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
