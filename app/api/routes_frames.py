from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from ..deps import get_frame_service
from ..services.frame_service import FrameService

router = APIRouter(prefix="", tags=["Frames"])

@router.post("/frames", status_code=202)
async def post_frames(file: UploadFile = File(default=None), svc: FrameService = Depends(get_frame_service)):
    try:
        img_bytes = await file.read()
        return await svc.process(img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error":"bad_request","message":str(e)})
