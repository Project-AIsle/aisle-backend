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

    async def process(self, payload: dict) -> dict:
        # Run detector
        detection = self.detector.infer(payload["frame"])
        if not detection:
            return {}
        return self.related_service(detection)
