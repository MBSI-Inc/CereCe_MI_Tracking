import asyncio
import re
import threading

from fastapi.routing import APIRoute, APIWebSocketRoute

from Gaze_Dashboard_Import.dashboard import application, runtime
from Gaze_Dashboard_Import.dashboard.paths import (
    CONFIG_PATH,
    STATIC_DIR,
    UNITY_BUILD_DIR,
)


def test_dashboard_assets_are_packaged():
    assert CONFIG_PATH.is_file()
    assert (STATIC_DIR / "index.html").is_file()
    assert (UNITY_BUILD_DIR / "WebGL_Build.loader.js").is_file()
    assert (UNITY_BUILD_DIR / "WebGL_Build.framework.js").is_file()
    assert (UNITY_BUILD_DIR / "WebGL_Build.data").is_file()
    assert (UNITY_BUILD_DIR / "WebGL_Build.wasm").is_file()


def test_browser_contract_has_unique_ids_and_expected_bridges():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    component_ids = re.findall(r'\bid=["\']([^"\']+)["\']', html)

    assert len(component_ids) == len(set(component_ids))
    assert "/ws/gaze" in html
    assert "/ws/camera" in html
    assert "GameManager" in html and "ReceiveGazeData" in html
    assert "WebGL_Build.loader.js" in html


def test_application_factory_registers_each_endpoint_once():
    app = application.create_app()
    routes = [
        (route.path, tuple(sorted(getattr(route, "methods", None) or [])))
        for route in app.routes
        if isinstance(route, (APIRoute, APIWebSocketRoute))
    ]

    expected = {
        ("/", ("GET",)),
        ("/api/state", ("GET",)),
        ("/api/calibrate/start", ("POST",)),
        ("/api/calibrate/reset", ("POST",)),
        ("/api/stop", ("POST",)),
        ("/api/recenter", ("POST",)),
        ("/api/config", ("POST",)),
        ("/static/unity/Build/{path:path}", ("GET",)),
        ("/ws/gaze", ()),
        ("/ws/camera", ()),
    }

    assert len(routes) == len(set(routes))
    assert expected.issubset(set(routes))


def test_import_does_not_start_dashboard_worker():
    names = {thread.name for thread in threading.enumerate()}
    assert "gaze-dashboard-webcam" not in names
    assert runtime.tracker is None


def test_unity_route_uses_webassembly_content_type():
    app = application.create_app()
    unity_route = next(
        route
        for route in app.routes
        if getattr(route, "path", "") == "/static/unity/Build/{path:path}"
    )

    response = asyncio.run(unity_route.endpoint("WebGL_Build.wasm"))

    assert response.media_type == "application/wasm"


def test_lifespan_starts_and_stops_dashboard_worker():
    app = application.create_app()

    async def exercise_lifespan():
        async with app.router.lifespan_context(app):
            assert any(
                thread.name == "gaze-dashboard-webcam" and thread.is_alive()
                for thread in threading.enumerate()
            )

    asyncio.run(exercise_lifespan())
    assert not any(
        thread.name == "gaze-dashboard-webcam" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_stop_releases_tracker_resources():
    class Tracker:
        def __init__(self):
            self.stopped = False
            self.released = False

        def stop(self):
            self.stopped = True

        def deinit(self):
            self.released = True

    tracker = Tracker()
    runtime.tracker = tracker

    runtime.stop_tracking()

    assert tracker.stopped
    assert tracker.released
    assert runtime.tracker is None
