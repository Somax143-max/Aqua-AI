# AquaProtect-AI: Start Backend REST API (PowerShell)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\backend"
Write-Host "Starting AquaProtect-AI Backend API..." -ForegroundColor Cyan
python run_backend.py
