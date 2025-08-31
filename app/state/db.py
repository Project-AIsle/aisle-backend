# app/state/db.py

from __future__ import annotations
import asyncio
from typing import Optional, Tuple, List
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from ..config import settings
from pymongo.errors import OperationFailure


def _id_str(doc: dict | None):
    if not doc:
        return doc
    doc["id"] = str(doc.pop("_id"))
    return doc

class MongoState:
    def __init__(self):
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
        self.db: AsyncIOMotorDatabase = self.client[settings.mongodb_db]
        self.col_relateds = self.db["relateds"]
        self.col_items = self.db["items"]
        self.col_products = self.db["products"]

    async def get_db(self, db_name: str) -> AsyncIOMotorDatabase:
        return self.client[db_name]

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

    # ---- PRODUCTS ----
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


# -------- FastAPI dependencies --------

_state: Optional[MongoState] = None
_inited: bool = False
_lock = asyncio.Lock()

async def _ensure_indexes(state):
    async def ensure(col, keys, name, unique=False):
        spec_key = dict(keys) if isinstance(keys, (list, tuple)) else keys
        existing = [ix async for ix in col.list_indexes()]
        by_name = next((ix for ix in existing if ix.get("name") == name), None)
        by_key  = next((ix for ix in existing if dict(ix.get("key", {})) == spec_key), None)

        # Se já existe por chave:
        if by_key:
            # mesmo unique? então mantemos (não renomeia, Mongo não suporta rename)
            if bool(by_key.get("unique", False)) == bool(unique):
                return
            # unique diferente: dropar e recriar com opções corretas
            await col.drop_index(by_key["name"])
            await col.create_index(list(spec_key.items()), name=name, unique=unique)
            return

        # Se existe por nome mas com spec diferente: dropar
        if by_name and (dict(by_name.get("key", {})) != spec_key or bool(by_name.get("unique", False)) != bool(unique)):
            await col.drop_index(name)

        # Criar se não existir
        try:
            await col.create_index(list(spec_key.items()), name=name, unique=unique)
        except OperationFailure as e:
            # 85/86 -> já existe com outro nome: ignore (já garantido por by_key)
            if getattr(e, "code", None) not in (85, 86):
                raise

    await ensure(state.col_relateds, {"product":1, "related":1}, name="product_related_unique", unique=True)
    await ensure(state.col_items,    {"slug":1},                    name="item_slug_unique",      unique=True)
    await ensure(state.col_products, {"slug":1},                    name="product_slug_unique",   unique=True)


async def get_state() -> MongoState:
    global _state, _inited
    if _state is None:
        _state = MongoState()
    if not _inited:
        async with _lock:
            if not _inited:
                await _ensure_indexes(_state)
                _inited = True
    return _state

async def get_db() -> AsyncIOMotorDatabase:
    state = await get_state()
    return state.db
