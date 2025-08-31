from __future__ import annotations
import base64, os, json
from typing import List, Tuple, Dict
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None

from ..config import settings

class Detection:
    def __init__(self, product: str, confidence: float):
        self.product = product
        self.confidence = confidence

class Detector:
    def __init__(self):
        self.roi = settings.roi
        self.threshold = settings.match_threshold
        self.templates_dir = settings.templates_dir
        self.use_onnx = settings.use_onnx and ort is not None and os.path.exists(settings.onnx_model_path)
        self.templates: Dict[str, np.ndarray] = {}
        self.class_map: Dict[int, str] = {}
        self.session = None
        if self.use_onnx:
            self._init_onnx()
        else:
            self._load_templates()

    def _init_onnx(self):
        try:
            self.session = ort.InferenceSession(settings.onnx_model_path, providers=["CPUExecutionProvider"])
        except Exception:
            self.use_onnx = False
            self._load_templates()
            return
        if os.path.exists(settings.onnx_model_data):
            try:
                with open(settings.onnx_model_data, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.class_map = {int(x["id"]): x["productId"] for x in data.get("classes", [])}
            except Exception:
                self.class_map = {}

    def _load_templates(self):
        self.templates.clear()
        if not os.path.isdir(self.templates_dir):
            return
        for f in os.listdir(self.templates_dir):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                pid = os.path.splitext(f)[0]
                img = cv2.imread(os.path.join(self.templates_dir, f), cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[pid] = img

    @staticmethod
    def _decode_b64(b64: str):
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _load_url(url: str):
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:
            data = r.read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _crop_roi(self, img):
        x1,y1,x2,y2 = self.roi
        h,w = img.shape[:2]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            return img, (0,0,w-1,h-1)
        return img[y1:y2, x1:x2].copy(), (x1,y1,x2,y2)

    def infer(self, frame_dict: dict) -> List[Detection]:
        enc = frame_dict.get("encoding")
        if enc == "base64":
            img = self._decode_b64(frame_dict.get("data",""))
        elif enc == "url":
            img = self._load_url(frame_dict.get("url",""))
        else:
            raise ValueError("Unsupported frame encoding")
        if img is None:
            return []
        roi_img, _ = self._crop_roi(img)
        if self.use_onnx and self.session:
            return self._infer_onnx(roi_img)
        return self._infer_templates(roi_img)

    def _infer_templates(self, roi_img) -> List[Detection]:
        out: List[Detection] = []
        if not self.templates or roi_img.size == 0:
            return out
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        for pid, tmpl in self.templates.items():
            th, tw = tmpl.shape[:2]
            if th >= roi_img.shape[0] or tw >= roi_img.shape[1]:
                continue
            tgray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, tgray, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, _ = cv2.minMaxLoc(res)
            if float(maxv) >= self.threshold:
                out.append(Detection(product=pid, confidence=float(maxv)))
        return out

    def _infer_onnx(self, roi_img) -> List[Detection]:
        # Expect output [N,6]: x1,y1,x2,y2,score,cls
        target = 640
        h,w = roi_img.shape[:2]
        scale = min(target/h, target/w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(roi_img, (nw, nh))
        pad = np.full((target, target, 3), 114, dtype=np.uint8)
        pad[:nh, :nw] = resized
        inp = pad.transpose(2,0,1)[None].astype(np.float32)/255.0
        name = self.session.get_inputs()[0].name
        preds = self.session.run(None, {name: inp})[0]
        out: List[Detection] = []
        for _,_,_,_,score,cls in preds:
            score = float(score)
            if score < self.threshold:
                continue
            product = self.class_map.get(int(cls), f"class-{int(cls)}")
            out.append(Detection(product=product, confidence=score))
        return out
