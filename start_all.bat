@echo off
title AquaProtect-AI Dual Service Launcher
echo ======================================================================
echo   AquaProtect-AI: Autonomous Marine Debris Command Center
echo   National Institute of Ocean Technology (NIOT) / MoES
echo ======================================================================
echo Launching Backend REST API service...
start "AquaProtect-AI Backend API" cmd /k "cd /d %~dp0backend && python run_backend.py"
echo Waiting for backend service to initialize...
timeout /t 3 /nobreak >nul
echo Launching Frontend Streamlit UI Dashboard...
start "AquaProtect-AI Frontend UI" cmd /k "cd /d %~dp0frontend && python run_frontend.py"
echo.
echo [SUCCESS] Both Backend (FastAPI) and Frontend (Streamlit) are now running!
echo Access the UI at: http://localhost:8501
echo Access the API at: http://localhost:8000/docs
