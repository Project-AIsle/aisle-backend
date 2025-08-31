from __future__ import annotations
from datetime import datetime, timezone
from typing import List

from ..core.detector import Detector
from ..state.db import MongoState

class FrameService:
    def __init__(self, detector: Detector, state: MongoState):
        self.detector = detector
        self.state = state

    async def process(self, payload: dict) -> dict:
        # Run detector
        detections = self.detector.infer(payload["frame"])
        detected_products = [ {"product": d.product, "confidence": d.confidence} for d in detections ]

        # Gather relateds for products in cart + detected
        cart = payload.get("cartProductIds") or []
        keys = set(cart + [d["product"] for d in detected_products])
        suggested: list[dict] = []
        for p in keys:
            rels, _ = await self.state.list_relateds(page=1, limit=999, product=p)
            suggested.extend(rels)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "id": f"frm_{int(datetime.now().timestamp())}",
            "receivedAt": now,
            "status": "queued",
            "detected": detected_products,
            "suggested": suggested
        }
