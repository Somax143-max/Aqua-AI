# AquaProtect-AI: Start Both Backend and Frontend Services (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  AquaProtect-AI: Autonomous Marine Debris Command Center" -ForegroundColor Yellow
Write-Host "  National Institute of Ocean Technology (NIOT) / MoES" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "Launching Backend REST API in a new window..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ScriptDir\backend'; python run_backend.py"

Start-Sleep -Seconds 3

Write-Host "Launching Frontend Dashboard in a new window..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ScriptDir\frontend'; python run_frontend.py"

Write-Host "`n[SUCCESS] Both Backend (FastAPI) and Frontend (Streamlit) are now running!" -ForegroundColor Green
Write-Host "Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "API Docs:  http://localhost:8000/docs (or http://localhost:8001/docs)" -ForegroundColor White
