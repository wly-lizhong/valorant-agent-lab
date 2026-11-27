@echo off
cd /d "%~dp0"
python scripts\start_windows.py
if errorlevel 1 (
  echo.
  echo Startup failed. Keep this window and send the error message.
  pause
)
