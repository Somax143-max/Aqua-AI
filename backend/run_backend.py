"""
AquaProtect-AI: Backend Server Launcher
National Institute of Ocean Technology (NIOT) / MoES
Smart India Hackathon (SIH26057)

Launches the FastAPI REST Service with dynamic port detection and graceful logging.
"""

import os
import sys
import socket
import uvicorn

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def find_available_port(default_port: int = 8000, fallback_ports=(8001, 8002, 8088, 8888)) -> int:
    env_port = os.environ.get("PORT") or os.environ.get("BACKEND_PORT")
    if env_port:
        return int(env_port)
    
    if not is_port_in_use(default_port):
        return default_port
    
    for port in fallback_ports:
        if not is_port_in_use(port):
            return port
            
    return default_port

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = find_available_port()
    
    print("=" * 70)
    print("  AquaProtect-AI: Autonomous Marine Command Center - Backend API")
    print(f"  Listening on: http://{host}:{port}")
    print(f"  Interactive API Docs: http://localhost:{port}/docs")
    print("=" * 70)
    
    uvicorn.run("api:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
