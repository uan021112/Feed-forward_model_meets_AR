from dotenv import load_dotenv

load_dotenv()

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _model_preload_enabled(name: str) -> bool:
    return os.environ.get(f"{name}_PRELOAD_ON_STARTUP", "1").lower() not in {"0", "false", "no"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if _model_preload_enabled("IGGT"):
        from app.services.iggt import preload_iggt_model
        preload_iggt_model()
    if _model_preload_enabled("LSEG"):
        from app.services.semantic import preload_lseg_model
        preload_lseg_model()
    try:
        yield
    finally:
        from app.services.iggt import unload_iggt_model
        from app.services.semantic import unload_lseg_model
        unload_iggt_model()
        unload_lseg_model()


app = FastAPI(
    title="ICXR 3D Reconstruction API",
    description="IGGT-based on-the-fly mobile AR tour deployment service",
    version="0.2.0",
    lifespan=lifespan,
)

from app.routers import reconstruct  # noqa: E402, F401

app.include_router(reconstruct.router, prefix="/api/v1")
