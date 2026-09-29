@echo off
title AquaProtect-AI Backend REST API
echo ===================================================
echo Starting AquaProtect-AI Backend REST Service (FastAPI)
echo ===================================================
cd /d "%~dp0backend"
python run_backend.py
pause
