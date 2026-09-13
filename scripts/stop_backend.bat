@echo off
setlocal
REM Stop the Ultron backend.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend.ps1" -Action stop
