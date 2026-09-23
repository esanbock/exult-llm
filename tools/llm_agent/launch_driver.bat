@echo off
rem Fully-detached driver launcher. Starts python in its own window with no
rem inherited stdout/stderr handles, so the caller returns immediately.
cd /d "%~dp0"
start "llm-driver" /min cmd /c "python -u driver.py --model gemma4:latest --show-thoughts --auto-feed --auto-feed-every 15 --save-every 40 --num-ctx 16384 --steps 20000 --delay 2.0 > driver_log.txt 2> driver_err.txt"
exit /b 0
