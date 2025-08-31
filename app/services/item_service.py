from __future__ import annotations
from typing import Optional
from ..state.db import MongoState

class ItemService:
    def __init__(self, state: MongoState):
        self.state = state

    async def create(self, data: dict) -> dict:
        return await self.state.create_item(data)

    async def list(self, page: int, limit: int, q: Optional[str]=None):
        return await self.state.list_items(page, limit, q)

    async def delete(self, id: str) -> bool:
        return await self.state.delete_item(id)
