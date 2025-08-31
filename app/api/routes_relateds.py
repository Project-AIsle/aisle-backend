# app/api/routes_relateds.py  (substitua pelo abaixo)

from typing import List, Union, Optional
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from fastapi.responses import HTMLResponse

from app.state.db import get_db
from app.services.related_service import RelatedService
from app.api.schemas import RelatedUpsert, RelatedUpsertBatch
from app.deps import get_related_service

router = APIRouter()

@router.get("/relateds")
async def list_relateds(
    product: Optional[str] = None,
    related: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
    svc: AsyncIOMotorDatabase = Depends(get_related_service),
):
    return await svc.list_relateds(product, related, page, limit)

@router.put("/relateds")
async def upsert_related(
    payload: Union[RelatedUpsert, RelatedUpsertBatch, List[RelatedUpsert]],
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    svc = RelatedService(db)
    if isinstance(payload, list):
        items = [p.model_dump(exclude_none=True) for p in payload]
    elif isinstance(payload, RelatedUpsertBatch):
        items = [p.model_dump(exclude_none=True) for p in payload.items]
    else:
        items = [payload.model_dump(exclude_none=True)]

    out = []
    for it in items:
        out.append(await svc.upsert_partial(it))
    return {"items": out, "count": len(out)}

@router.delete("/relateds/{id}")
async def delete_related(id: str, db: AsyncIOMotorDatabase = Depends(get_db)):
    try:
        oid = ObjectId(id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid id")
    res = await db["relateds"].delete_one({"_id": oid})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="not found")
    return {"deleted": True}


@router.get("/relateds/suggest", response_class=HTMLResponse)
async def suggest_related_item_html(
    product: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    svc = RelatedService(db)
    html_str = await svc.pick_related_item(product)  # agora retorna HTML
    return HTMLResponse(content=html_str, media_type="text/html")