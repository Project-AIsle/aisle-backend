from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from .schemas import FrameInput, FrameAccepted
from ..deps import get_frame_service
from ..services.frame_service import FrameService

router = APIRouter(prefix="", tags=["Frames"])

@router.post("/frames", response_model=FrameAccepted, status_code=202)
async def post_frames(payload: FrameInput, svc: FrameService = Depends(get_frame_service)):
    try:
        return await svc.process(payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error":"bad_request","message":str(e)})
