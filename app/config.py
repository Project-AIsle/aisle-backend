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

def _tuple4(txt: str | None, default: str) -> Tuple[int,int,int,int]:
    raw = (txt or default).split(",")
    vals = [int(x.strip()) for x in raw]
    if len(vals) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    return (vals[0], vals[1], vals[2], vals[3])

@dataclass
class Settings:
    api_prefix: str = os.getenv("API_PREFIX", "/v1")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Mongo
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "narvalcart")
    # CV
    roi: Tuple[int,int,int,int] = _tuple4(os.getenv("ROI"), "100,100,400,400")
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", "0.78"))
    templates_dir: str = os.getenv("TEMPLATES_DIR", "app/assets/templates")
    use_onnx: bool = os.getenv("USE_ONNX", "false").lower() in {"1","true","yes"}
    onnx_model_path: str = os.getenv("ONNX_MODEL_PATH", "app/assets/models/detection.onnx")
    onnx_model_data: str = os.getenv("ONNX_MODEL_DATA", "app/assets/models/model.data")
    # upload dir
    upload_dir: str = str(Path(__file__).resolve().parent / "assets" / "uploads")
    public_base_url: str = "http://localhost:8080"   # <<<<<< ABSOLUTE BASE

    class Config:
        env_file = ".env"
        extra = "ignore"
@lru_cache
def _get() -> "Settings":
    s = Settings()
    Path(s.upload_dir).mkdir(parents=True, exist_ok=True)
    return s


settings = Settings()
