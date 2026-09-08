"""Application factory for the gaze dashboard."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import runtime
from .paths import STATIC_DIR, UNITY_BUILD_DIR
from .routes import create_dashboard_router
from .unity_assets import create_unity_router


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    runtime.start()
    try:
        yield
    finally:
        runtime.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Cerebruh Gaze Dashboard", lifespan=_lifespan)
    app.include_router(create_unity_router(UNITY_BUILD_DIR))
    app.include_router(create_dashboard_router())
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
