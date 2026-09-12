"""Standalone entry point for the gaze dashboard."""

import uvicorn

if __package__:
    from .dashboard import create_app
else:
    from dashboard import create_app


app = create_app()


if __name__ == "__main__":
    print("=" * 56)
    print("  Cerebruh Gaze Dashboard")
    print("  http://127.0.0.1:8050")
    print("=" * 56)
    uvicorn.run(app, host="127.0.0.1", port=8050, log_level="warning")
