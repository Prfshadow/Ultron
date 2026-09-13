@echo off
setlocal
REM Start the Ultron backend in the background (headless, with logs).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend.ps1" -Action start
