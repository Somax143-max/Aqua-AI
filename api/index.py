"""
AquaProtect-AI: Vercel Serverless Function Handler
Exposes the FastAPI Backend API on Vercel's serverless runtime (@vercel/python).
"""

import sys
import os

# Resolve absolute paths to ensure all backend and core modules import smoothly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

for path in [ROOT_DIR, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend.api import app
except ImportError:
    from api import app

# Vercel ASGI application handler
handler = app
