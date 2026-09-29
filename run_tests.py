"""
AquaProtect-AI: Automated Test Runner
Runs the full suite of 25 unit and integration tests across hydroacoustics and backend API.
"""

import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "pytest", "backend/tests/", "-v"]
    print(f"Running test suite: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)
