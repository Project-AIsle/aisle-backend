from __future__ import annotations
from fastapi import FastAPI
from .config import settings
from .logging_conf import configure_logging
from .api.routes_relateds import router as relateds_router
from .api.routes_items import router as items_router
from .api.routes_frames import router as frames_router
from fastapi.staticfiles import StaticFiles
from pathlib import Path



configure_logging()

app = FastAPI(title="Narval Cart API", version="1.0.0")

api = FastAPI(title="Narval Cart API - v1", version="1.0.0", docs_url=None, redoc_url=None)
api.include_router(relateds_router)
api.include_router(items_router)
api.include_router(frames_router)

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

app.mount(settings.api_prefix, api)

upload_dir = Path(getattr(settings, "upload_dir", "app/assets/uploads")).resolve()
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(upload_dir)), name="static")