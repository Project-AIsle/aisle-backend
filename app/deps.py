from __future__ import annotations
from functools import lru_cache

from .state.db import MongoState, get_db
from .core.detector import Detector, YOLOXDetector, CLIPClassifier
from .services.related_service import RelatedService
from .services.item_service import ItemService
from .services.frame_service import FrameService
from .config import settings
import asyncio

@lru_cache(maxsize=1)
def get_state() -> MongoState:
    return MongoState()

@lru_cache(maxsize=1)
async def get_detector() -> Detector:
    svc = await get_related_service()
    all_items = asyncio.run(svc.list_relateds())
    all_classes = list({item["related"] for item in all_items["items"]})

    yolo = YOLOXDetector(model_path=settings.YOLOX_MODEL, providers=[settings.ORT_PROVIDERS], input_size=(settings.YOLOX_INPUT_W, settings.YOLOX_INPUT_H), score_thr=settings.YOLOX_CONF_THR, iou_thr=settings.YOLOX_IOU_THR)
    classifier = CLIPClassifier(model_name=settings.CLIP_MODEL_NAME, device=settings.CLIP_DEVICE)
    return Detector(yolo_detector=yolo, clip_classifier=classifier, classes=all_classes)

@lru_cache(maxsize=1)
async def get_related_service() -> RelatedService:
    db = await get_db()
    return RelatedService(db)

@lru_cache(maxsize=1)
async def get_item_service() -> ItemService:
    db = await get_db()
    return ItemService(db)

@lru_cache(maxsize=1)
async def get_frame_service() -> FrameService:
    detector = await get_detector()
    related_service = await get_related_service()
    return FrameService(detector, get_state(), related_service)
