from __future__ import annotations
from datetime import datetime, timezone
from typing import List

from .related_service import RelatedService
from ..core.detector import Detector
from ..state.db import MongoState

class FrameService:
    def __init__(self, detector: Detector, state: MongoState, related_service: RelatedService):
        self.detector = detector
        self.state = state
        self.related_service = related_service

    async def process(self, payload: bytes) -> str:
        # Run detector
        detection = self.detector.infer(payload)
        if not detection:
            return ""
        return await self.related_service.pick_related_item(detection)
