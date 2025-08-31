# app/services/related_service.py
from datetime import datetime

class RelatedService:
    def __init__(self, db):
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
