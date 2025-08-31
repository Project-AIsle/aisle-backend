from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from ..deps import get_frame_service
from fastapi.responses import HTMLResponse
from ..services.frame_service import FrameService

router = APIRouter(prefix="", tags=["Frames"])

@router.post("/frames", status_code=200, response_class=HTMLResponse)
async def post_frames(file: UploadFile = File(default=None)):
    try:
        svc = await get_frame_service()
        img_bytes = await file.read()
        return await svc.process(img_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error":"bad_request","message":str(e)})
