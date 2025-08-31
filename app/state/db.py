from __future__ import annotations
from typing import Optional, Tuple, List
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from ..config import settings

def _id_str(doc: dict | None):
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    return doc

class MongoState:
    def __init__(self):
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db]
        self.col_relateds = self.db["relateds"]
        self.col_items = self.db["items"]
        self.col_products = self.db["products"]  # referenced collection

    # ---- RELATEDS ----
    async def create_relateds(self, docs: List[dict]) -> list[dict]:
        created = []
        for d in docs:
            exists = await self.col_relateds.find_one({"product": d["product"], "related": d["related"]})
            if exists:
                continue
            res = await self.col_relateds.insert_one(d)
            cur = await self.col_relateds.find_one({"_id": res.inserted_id})
            created.append(_id_str(cur))
        return created

    async def list_relateds(self, page: int, limit: int, product: Optional[str]=None, related: Optional[str]=None) -> Tuple[list[dict], int]:
        query: dict = {}
        if product:
            query["product"] = product
        if related:
            query["related"] = related
        total = await self.col_relateds.count_documents(query)
        cursor = self.col_relateds.find(query).skip((page-1)*limit).limit(limit).sort("_id", -1)
        data = [_id_str(d) async for d in cursor]
        return data, total

    async def delete_related(self, id: str) -> bool:
        res = await self.col_relateds.delete_one({"_id": ObjectId(id)})
        return res.deleted_count > 0

    # ---- ITEMS ----
    async def create_item(self, data: dict) -> dict:
        res = await self.col_items.insert_one(data)
        doc = await self.col_items.find_one({"_id": res.inserted_id})
        return _id_str(doc)

    async def list_items(self, page: int, limit: int, q: Optional[str]=None) -> Tuple[list[dict], int]:
        query: dict = {}
        if q:
            query = {"name": {"$regex": q, "$options": "i"}}
        total = await self.col_items.count_documents(query)
        cursor = self.col_items.find(query).skip((page-1)*limit).limit(limit).sort("_id", -1)
        data = [_id_str(d) async for d in cursor]
        return data, total

    async def delete_item(self, id: str) -> bool:
        res = await self.col_items.delete_one({"_id": ObjectId(id)})
        return res.deleted_count > 0

    # ---- PRODUCTS (support collection) ----
    async def create_product(self, data: dict) -> dict:
        res = await self.col_products.insert_one(data)
        doc = await self.col_products.find_one({"_id": res.inserted_id})
        return _id_str(doc)

    async def list_products(self, page: int, limit: int, q: Optional[str]=None) -> Tuple[list[dict], int]:
        query: dict = {}
        if q:
            query = {"$or":[{"type":{"$regex":q,"$options":"i"}},{"name":{"$regex":q,"$options":"i"}}]}
        total = await self.col_products.count_documents(query)
        cursor = self.col_products.find(query).skip((page-1)*limit).limit(limit).sort("_id", -1)
        data = [_id_str(d) async for d in cursor]
        return data, total

    async def delete_product(self, id: str) -> bool:
        res = await self.col_products.delete_one({"_id": ObjectId(id)})
        return res.deleted_count > 0
