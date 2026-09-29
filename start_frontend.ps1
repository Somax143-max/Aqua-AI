# AquaProtect-AI: Start Frontend Dashboard (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\frontend"
Write-Host "Starting AquaProtect-AI Frontend Dashboard..." -ForegroundColor Cyan
python run_frontend.py
