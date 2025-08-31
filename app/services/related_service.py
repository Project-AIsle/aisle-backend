# app/services/related_service.py
import re
from datetime import datetime
from typing import Optional, List
from unidecode import unidecode
from motor.motor_asyncio import AsyncIOMotorDatabase
from jinja2 import Environment, FileSystemLoader, select_autoescape
from ..config import settings
from pathlib import Path
import os, re


UPLOAD_DIR = getattr(settings, "upload_dir", "app/assets/uploads")
BASE_URL = getattr(settings, "public_base_url", "").rstrip("/")
TEMPLATES_DIR = Path("app/templates")

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

def _slugify(text: str) -> str:
    text = unidecode(text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "item"

def _to_static_url(path: str) -> str:
    try:
        up = Path(UPLOAD_DIR).resolve()
        p  = Path(path).resolve()
        rel = p.relative_to(up)
        # ABSOLUTE: http://localhost:8080/static/...
        return f"{BASE_URL}/static/{str(rel).replace(os.sep,'/')}"
    except Exception:
        return ""


class RelatedService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.col = db["relateds"]

    async def list_relateds(
        self,
        product: Optional[str] = None,
        related: Optional[str] = None,
        page: int = 1,
        limit: int = 50,
    ):
        q: dict = {}
        if product:
            q["product"] = product
        if related:
            q["related"] = related

        total = await self.db["relateds"].count_documents(q)
        cursor = self.db["relateds"].find(q).skip((page - 1) * limit).limit(limit).sort("_id", -1)

        items = []
        async for d in cursor:   # <<< motor: usar async for
            d["id"] = str(d.pop("_id"))
            items.append(d)

        return {"items": items, "total": total, "page": page, "limit": limit}

    async def upsert_partial(self, payload: dict) -> dict:
        product = payload["product"]
        related = payload["related"]
        update = {"$setOnInsert": {"product": product, "related": related, "createdAt": datetime.utcnow()}}
        if "score" in payload and payload["score"] is not None:
            update.setdefault("$set", {})["score"] = payload["score"]
        await self.col.update_one({"product": product, "related": related}, update, upsert=True)
        doc = await self.col.find_one({"product": product, "related": related})
        doc["id"] = str(doc.pop("_id"))
        return doc

    async def pick_related_item(self, product: str) -> str:
        """
        Fluxo completo:
        - escolhe 1 related aleatório para 'product'
        - tenta achar item correspondente em 'items' (por slug/name)
        - RENDERIZA e RETORNA HTML (Jinja2):
            * se encontrou item -> item_card.html (com imagem se houver)
            * se não encontrou -> promo.html ("Compre {related}")
        """
        cur = self.col.aggregate([
            {"$match": {"product": product}},
            {"$sample": {"size": 1}},
        ])
        docs = await cur.to_list(length=1)
        related = (docs[0].get("related") if docs else None)

        if related:
            slug = _slugify(related)
            item = await self.db["items"].find_one({
                "$or": [
                    {"slug": slug},
                    {"name": {"$regex": f"^{re.escape(related)}$", "$options": "i"}},
                ]
            })
        else:
            item = None

        if item:
            ctx = {
                "product": product,
                "has_item": True,
                "item": {
                    "id": str(item["_id"]),
                    "name": item.get("name") or related,
                    "slug": item.get("slug") or slug,
                    "img_url": _to_static_url(item.get("path", "")),
                },
            }
            tpl = env.get_template("item_card.html")
            return tpl.render(**ctx)

        ctx = {"product": product, "has_item": False, "related": related}
        tpl = env.get_template("promo.html")
        return tpl.render(**ctx)
