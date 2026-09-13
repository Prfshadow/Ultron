@echo off
setlocal
REM Remove the Ultron auto-start launcher from the Startup folder.
REM (Does NOT stop a running backend.)

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
del /Q "%STARTUP%\ultron_backend.vbs" 2>nul

echo Auto-start removed. The backend keeps running until you stop it with:
echo   scripts\stop_backend.bat
