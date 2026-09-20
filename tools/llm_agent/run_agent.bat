@echo off
cd /d "%~dp0"
python driver.py --model gemma4:latest --show-thoughts --auto-feed --auto-feed-every 15 --steps 400 --delay 2.5
echo.
echo === driver exited (code %ERRORLEVEL%) ===
pause
