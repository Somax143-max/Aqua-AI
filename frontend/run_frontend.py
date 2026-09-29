"""
AquaProtect-AI: Frontend Dashboard Launcher
National Institute of Ocean Technology (NIOT) / MoES
Smart India Hackathon (SIH26057)

Launches the Streamlit Maritime Command Center Dashboard.
"""

import os
import sys
import subprocess

def main():
    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(frontend_dir, "app.py")
    port = os.environ.get("PORT") or os.environ.get("FRONTEND_PORT", "8501")
    
    print("=" * 70)
    print("  AquaProtect-AI: Marine Command Center UI Dashboard")
    print(f"  Launching Streamlit UI on: http://localhost:{port}")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        app_path,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--theme.base=light"
    ]
    
    try:
        subprocess.run(cmd, cwd=frontend_dir, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Frontend dashboard stopped by user.")
    except Exception as exc:
        print(f"[ERROR] Failed to run Streamlit frontend: {exc}")

if __name__ == "__main__":
    main()
