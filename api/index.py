"""Vercel serverless entry for the FastAPI live lab."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_server import app  # noqa: E402

# Vercel Python runtime serves this ASGI app.
__all__ = ["app"]
