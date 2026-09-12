# Gaze dashboard

This directory contains the gaze dashboard migrated from
`Cerebruh_Gaze_Tracking`. The browser UI and bundled Unity WebGL game are
served by FastAPI. The gaze tracker remains the source of camera and gaze
data.

## Run locally

From the repository root:

```powershell
conda activate BCI-Gazetrack
python -m pip install -r requirements.txt
python Gaze_Dashboard_Import\app.py
```

Open <http://127.0.0.1:8050/>. The camera is opened when calibration starts
and released by **STOP** or application shutdown.

The `websockets` dependency supplies the transport used by `/ws/gaze` and
`/ws/camera`. FastAPI and Uvicorn can serve the HTML page without it, but
Uvicorn will reject those upgrade requests.

## Use from another ASGI application

The application factory can be imported without starting the server, camera,
or background workers:

```python
from Gaze_Dashboard_Import import create_app

gaze_dashboard = create_app()
```

FastAPI starts the frame worker through its lifespan hook and closes tracker
resources during shutdown. The public browser endpoints remain:

- `/api/state`, `/api/calibrate/start`, `/api/calibrate/reset`
- `/api/stop`, `/api/recenter`, `/api/config`
- `/ws/gaze`, `/ws/camera`
- `/static/unity/Build/...`

The configuration used by calibration and tracking is
`Gaze_Dashboard_Import/config.json`.

## Structure

```text
Gaze_Dashboard_Import/
├── app.py                       standalone launcher
├── config.json                  gaze and controller settings
├── facemesh_gazetracker.py      MediaPipe gaze source
├── util.py                      gaze calculation utilities
├── unity_connect.py             legacy Unity socket adapter
├── dashboard/
│   ├── application.py           application factory and lifespan
│   ├── routes.py                HTTP and WebSocket transport
│   ├── runtime.py               tracker, calibration and frame state
│   ├── paths.py                 package-relative paths
│   └── unity_assets.py          Unity WebGL file serving
└── static/
    ├── index.html               dashboard UI and browser/Unity bridge
    └── unity/Build/             bundled WebGL build
```

The browser-based Unity bridge in `static/index.html` is the active game
integration. `unity_connect.py` is retained for compatibility with gaze
tracker modes that still use the older socket path.
