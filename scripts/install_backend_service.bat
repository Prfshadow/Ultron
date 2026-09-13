@echo off
setlocal
REM Make the Ultron backend start automatically at Windows logon.
REM Uses the user's Startup folder, so NO administrator rights are needed.
REM The launcher (scripts\ultron_backend.vbs) runs headless and logs to .\logs.

set "VBS=%~dp0ultron_backend.vbs"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"

copy /Y "%VBS%" "%STARTUP%\ultron_backend.vbs" >nul
if errorlevel 1 (
    echo Failed to copy the launcher into the Startup folder.
    exit /b 1
)

echo.
echo Installed: Ultron backend will start automatically when you log in.
echo   - To start it right now, run:   scripts\start_backend.bat
echo   - To stop it:                   scripts\stop_backend.bat
echo   - To check it:                  scripts\backend_status.bat
echo   - To remove auto-start:         scripts\uninstall_backend_service.bat
