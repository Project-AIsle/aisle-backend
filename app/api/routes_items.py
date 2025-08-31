# app/api/routes_items.py
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from app.state.db import get_db
from app.services.item_service import ItemService

router = APIRouter()

@router.post("/items", status_code=status.HTTP_201_CREATED)
async def upsert_item(
    name: str = Form(...),
    file: UploadFile = File(...),
    db = Depends(get_db),
):
    data = await file.read()
    if not data:
        raise HTTPException(400, "file is empty")
    svc = ItemService(db)
    doc = await svc.upsert_from_upload(name=name, file_bytes=data, original_filename=file.filename)
    return doc

@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: str, db = Depends(get_db)):
    svc = ItemService(db)
    ok = await svc.delete(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="item not found")
    return None
