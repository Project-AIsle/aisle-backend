from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from .schemas import Related, RelatedInput, PaginatedRelated
from ..deps import get_related_service
from ..services.related_service import RelatedService

router = APIRouter(prefix="", tags=["Relateds"])

@router.post("/relateds", response_model=List[Related], status_code=201)
async def create_relateds(payload: List[RelatedInput] | RelatedInput,
                          svc: RelatedService = Depends(get_related_service)):
    docs = payload if isinstance(payload, list) else [payload]
    created = await svc.create_many([d.model_dump() for d in docs])
    if not created:
        raise HTTPException(status_code=409, detail={"error":"conflict","message":"Already exists"})
    return created

@router.get("/relateds", response_model=PaginatedRelated)
async def list_relateds(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    product: Optional[str] = None,
    related: Optional[str] = None,
    svc: RelatedService = Depends(get_related_service)
):
    data, total = await svc.list(page, limit, product, related)
    return {"data": data, "page": page, "limit": limit, "total": total}

@router.delete("/relateds/{id}", status_code=204)
async def delete_related(id: str, svc: RelatedService = Depends(get_related_service)):
    ok = await svc.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error":"not_found","message":"Related not found"})
    return None
