from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field
from typing import Optional, List

# Relateds collection
class RelatedInput(BaseModel):
    product: str = Field(description="Base product, e.g., 'macarrão'")
    related: str = Field(description="Related product, e.g., 'molho de tomate'")
    score: float = Field(ge=0, le=1, description="Relevance (0..1)")

class Related(RelatedInput):
    id: str

class PaginatedRelated(BaseModel):
    data: list[Related]
    page: int
    limit: int
    total: int

# Items (cart)
class ItemInput(BaseModel):
    name: str
    quantity: int = Field(ge=0)
    note: Optional[str] = None

class Item(ItemInput):
    id: str

class PaginatedItems(BaseModel):
    data: list[Item]
    page: int
    limit: int
    total: int

# Frames
class FrameImage(BaseModel):
    encoding: Literal["base64","url"]
    data: Optional[str] = None
    url: Optional[str] = None

class FrameInput(BaseModel):
    frame: FrameImage
    cartProductIds: Optional[list[str]] = None

class DetectionOut(BaseModel):
    product: str
    confidence: float

class FrameAccepted(BaseModel):
    id: str
    receivedAt: str
    status: Literal["queued","processing","done","failed"]
    detected: Optional[list[DetectionOut]] = None
    suggested: Optional[list[Related]] = None

class RelatedUpsert(BaseModel):
    product: str = Field(...)
    related: str = Field(...)
    score: Optional[float] = Field(None, ge=0, le=1)

class RelatedUpsertBatch(BaseModel):
    items: List[RelatedUpsert]