"""Filesystem locations used by the dashboard."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PACKAGE_ROOT / "static"
UNITY_BUILD_DIR = STATIC_DIR / "unity" / "Build"
CONFIG_PATH = PACKAGE_ROOT / "config.json"
