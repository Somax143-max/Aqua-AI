@echo off
title AquaProtect-AI Frontend Command Center
echo ===================================================
echo Starting AquaProtect-AI Frontend Dashboard (Streamlit)
echo ===================================================
cd /d "%~dp0frontend"
python run_frontend.py
pause
