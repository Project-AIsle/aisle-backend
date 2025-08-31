from __future__ import annotations
from typing import Optional, List, Tuple
from ..state.db import MongoState

class RelatedService:
    def __init__(self, state: MongoState):
        self.state = state

    async def create_many(self, docs: List[dict]) -> list[dict]:
        return await self.state.create_relateds(docs)

    async def list(self, page: int, limit: int, product: Optional[str]=None, related: Optional[str]=None):
        return await self.state.list_relateds(page, limit, product, related)

    async def delete(self, id: str) -> bool:
        return await self.state.delete_related(id)
