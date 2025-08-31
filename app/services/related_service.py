# app/services/related_service.py
import re
from datetime import datetime
from typing import Optional, List
from unidecode import unidecode
from motor.motor_asyncio import AsyncIOMotorDatabase

def _slugify(text: str) -> str:
    text = unidecode(text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text or "item"

class RelatedService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.col = db["relateds"]

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

    async def pick_related_item(self, product: str) -> dict:
        """
        Recebe um 'product', busca em 'relateds' por esse product,
        escolhe 1 relacionado aleatoriamente; tenta achar um item correspondente.
        - Se encontrar item: retorna o objeto do item (com 'id').
        - Se NÃO encontrar item: retorna {"related": "<string>"}
        - Se não houver relacionados: retorna {"related": None}
        """
        # escolhe 1 related aleatório no Mongo (mais eficiente que carregar todos)
        cur = self.col.aggregate([
            {"$match": {"product": product}},
            {"$sample": {"size": 1}},
        ])
        docs = await cur.to_list(length=1)
        if not docs:
            return {"related": None}

        related = docs[0].get("related")
        if not related:
            return {"related": None}

        slug = _slugify(related)

        item = await self.db["items"].find_one({
            "$or": [
                {"slug": slug},
                {"name": {"$regex": f"^{re.escape(related)}$", "$options": "i"}},
            ]
        })
        if item:
            item["id"] = str(item.pop("_id"))
            return item

        return {"related": related}
