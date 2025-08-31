from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache


load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "app/assets/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@dataclass
class Settings:
    api_prefix: str = os.getenv("API_PREFIX", "/v1")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Mongo
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://narvalcart-mongo:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "narvalcart")
    # CV
    # -----------------------------
    # ONNX Runtime / Modelos
    # -----------------------------
    # Detector (YOLO/YOLOX/YOLOv8 exportado para ONNX)
    YOLOX_MODEL: str = os.path.join(os.getcwd(), "app/assets/models/yolo/model.onnx")
    YOLOX_INPUT_W: int = 640
    YOLOX_INPUT_H: int = 640
    YOLOX_INPUT_NAME: str | None = None           # se None, detecta automaticamente
    YOLOX_CONF_THR: float = 0.20
    YOLOX_IOU_THR: float = 0.60

    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_DEVICE: str = "cpu"

    # Provedores do ONNX Runtime (separe por vírgulas para múltiplos)
    # Ex.: "CPUExecutionProvider"
    ORT_PROVIDERS: str = "CPUExecutionProvider"
 
    # Outros
    DEBUG_OUT_DIR: str = "debug_out"

    upload_dir: str = os.environ.get("UPLOAD_PATH")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
