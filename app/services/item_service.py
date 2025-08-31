# app/services/item_service.py
from datetime import datetime
import os, re, glob
from typing import Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from unidecode import unidecode
from ..config import settings

ITEMS_DIR = os.path.join(settings.upload_dir, "items")
os.makedirs(ITEMS_DIR, exist_ok=True)

def _id_str(doc: dict | None):
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    return doc

def _slugify(text: str) -> str:
    text = unidecode(text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "item"

class ItemService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.col = db["items"]

    def _target_path(self, slug: str, ext: str) -> str:
        return os.path.join(ITEMS_DIR, f"{slug}{ext}")

    def _purge_by_slug(self, slug: str, keep: Optional[str] = None):
        for p in glob.glob(os.path.join(ITEMS_DIR, f"{slug}.*")):
            if keep and os.path.abspath(p) == os.path.abspath(keep):
                continue
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    async def upsert_from_upload(self, *, name: str, file_bytes: bytes, original_filename: Optional[str]) -> dict:
        slug = _slugify(name)
        ext = ""
        if original_filename and "." in original_filename:
            ext = "." + original_filename.rsplit(".", 1)[-1].lower()
        if not ext:
            ext = ".bin"

        # localizar existente por SLUG ou NAME (case-insensitive)
        existing = await self.col.find_one({"$or": [
            {"slug": slug},
            {"name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
        ]})

        # se trocar de slug, remover imagens antigas do slug anterior
        old_slug = existing.get("slug") if existing else None
        if old_slug and old_slug != slug:
            self._purge_by_slug(old_slug)

        # gravar a imagem com slug final e garantir 1 arquivo por item
        os.makedirs(ITEMS_DIR, exist_ok=True)
        new_path = self._target_path(slug, ext)
        with open(new_path, "wb") as f:
            f.write(file_bytes)
        self._purge_by_slug(slug, keep=new_path)

        payload = {
            "name": name,
            "slug": slug,
            "path": new_path,
            "updatedAt": datetime.utcnow(),
        }

        if existing:
            await self.col.update_one({"_id": existing["_id"]}, {"$set": payload})
            doc = await self.col.find_one({"_id": existing["_id"]})
        else:
            payload["createdAt"] = datetime.utcnow()
            res = await self.col.insert_one(payload)
            doc = await self.col.find_one({"_id": res.inserted_id})

        return _id_str(doc)

    async def delete(self, item_id: str) -> bool:
        try:
            oid = ObjectId(item_id)
        except Exception:
            return False
        doc = await self.col.find_one({"_id": oid})
        if not doc:
            return False
        # remove imagem associada e variações por slug
        if doc.get("path") and os.path.isfile(doc["path"]):
            try: os.remove(doc["path"])
            except FileNotFoundError: pass
        if doc.get("slug"):
            self._purge_by_slug(doc["slug"])
        res = await self.col.delete_one({"_id": oid})
        return res.deleted_count == 1
