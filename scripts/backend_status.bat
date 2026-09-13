@echo off
setlocal
REM Show whether the Ultron backend is running.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend.ps1" -Action status
