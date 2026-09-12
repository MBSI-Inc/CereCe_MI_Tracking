"""Routes for the embedded Unity WebGL build."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

_BROTLI_MIME_TYPES = {
    ".wasm.br": "application/wasm",
    ".js.br": "application/javascript",
    ".data.br": "application/octet-stream",
    ".symbols.json.br": "application/octet-stream",
}

_PLAIN_MIME_TYPES = {
    ".wasm": "application/wasm",
    ".framework.js": "application/javascript",
    ".loader.js": "application/javascript",
    ".data": "application/octet-stream",
    ".js": "application/javascript",
}


def create_unity_router(build_directory: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/static/unity/Build/{path:path}")
    async def serve_unity_build(path: str):
        file_path = (build_directory / path).resolve()
        try:
            file_path.relative_to(build_directory.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Build file not found")

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"Build file not found: {path}")

        if path.endswith(".br"):
            media_type = _media_type(path, _BROTLI_MIME_TYPES)
            return FileResponse(
                path=str(file_path),
                media_type=media_type,
                headers={"Content-Encoding": "br"},
            )

        return FileResponse(
            path=str(file_path),
            media_type=_media_type(path, _PLAIN_MIME_TYPES),
        )

    return router


def _media_type(path: str, mime_types: dict) -> str:
    for suffix, media_type in mime_types.items():
        if path.endswith(suffix):
            return media_type
    return "application/octet-stream"
