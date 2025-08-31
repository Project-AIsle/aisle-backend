from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from .schemas import Item, ItemInput, PaginatedItems
from ..deps import get_item_service
from ..services.item_service import ItemService

router = APIRouter(prefix="", tags=["Items"])

@router.post("/items", response_model=Item, status_code=201)
async def create_item(payload: ItemInput, svc: ItemService = Depends(get_item_service)):
    return await svc.create(payload.model_dump())

@router.get("/items", response_model=PaginatedItems)
async def list_items(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    q: Optional[str] = None,
    svc: ItemService = Depends(get_item_service)
):
    data, total = await svc.list(page, limit, q)
    return {"data": data, "page": page, "limit": limit, "total": total}

@router.delete("/items/{id}", status_code=204)
async def delete_item(id: str, svc: ItemService = Depends(get_item_service)):
    ok = await svc.delete(id)
    if not ok:
        raise HTTPException(status_code=404, detail={"error":"not_found","message":"Item not found"})
    return None
