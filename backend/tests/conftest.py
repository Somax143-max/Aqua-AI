"""
Pytest configuration for AquaProtect-AI backend test suite.
Automatically ensures backend directory is discoverable on sys.path.
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
