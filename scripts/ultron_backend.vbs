' Ultron backend auto-start launcher (hidden, at Windows logon).
' Installed into the user's Startup folder by scripts\install_backend_service.bat
Set sh = CreateObject("WScript.Shell")
root = "C:\Users\avina\Desktop\Projects\Ultron"
sh.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & root & "\scripts\backend.ps1"" -Action start", 0, False
