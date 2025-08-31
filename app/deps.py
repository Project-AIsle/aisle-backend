from __future__ import annotations
from functools import lru_cache

from .state.db import MongoState
from .core.detector import Detector
from .services.related_service import RelatedService
from .services.item_service import ItemService
from .services.frame_service import FrameService

@lru_cache(maxsize=1)
def get_state() -> MongoState:
    return MongoState()

@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()

@lru_cache(maxsize=1)
def get_related_service() -> RelatedService:
    return RelatedService(get_state())

@lru_cache(maxsize=1)
def get_item_service() -> ItemService:
    return ItemService(get_state())

@lru_cache(maxsize=1)
def get_frame_service() -> FrameService:
    return FrameService(get_detector(), get_state())
