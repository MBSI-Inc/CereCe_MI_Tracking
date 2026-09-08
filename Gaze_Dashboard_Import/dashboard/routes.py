"""HTTP and WebSocket routes for the gaze dashboard."""

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import runtime
from .paths import STATIC_DIR


def create_dashboard_router() -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @router.get("/api/state")
    async def get_state():
        return JSONResponse(runtime.snapshot())

    @router.post("/api/calibrate/start")
    async def calibrate_start():
        runtime.start_calibration()
        return JSONResponse({"ok": True, "step": 1})

    @router.post("/api/calibrate/reset")
    async def calibrate_reset():
        runtime.reset_calibration()
        return JSONResponse({"ok": True, "step": 0})

    @router.post("/api/stop")
    async def stop_tracking():
        runtime.stop_tracking()
        return JSONResponse({"ok": True})

    @router.post("/api/recenter")
    async def recenter():
        runtime.recenter()
        return JSONResponse({"ok": True})

    @router.post("/api/config")
    async def update_config(request: Request):
        runtime.update_configuration(await request.json())
        return JSONResponse({"ok": True})

    @router.websocket("/ws/gaze")
    async def gaze_socket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_text(
                    json.dumps(runtime.next_gaze_payload(), default=str)
                )
                await asyncio.sleep(0.15)
        except WebSocketDisconnect:
            return

    @router.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                frame = runtime.latest_camera_frame()
                if frame:
                    await websocket.send_text(frame)
                await asyncio.sleep(0.06)
        except WebSocketDisconnect:
            return

    return router
