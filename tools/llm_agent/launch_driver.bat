@echo off
rem Fully-detached driver launcher. Starts python in its own window with no
rem inherited stdout/stderr handles, so the caller returns immediately.
rem Usage: launch_driver.bat [model]   (default: gemma4:latest)
cd /d "%~dp0"
set MODEL=%1
if "%MODEL%"=="" set MODEL=gemma4:latest
start "llm-driver" /min cmd /c "python -u driver.py --model %MODEL% --show-thoughts --auto-feed --auto-feed-every 15 --save-every 40 --num-ctx 32768 --steps 20000 --delay 2.0 > driver_log.txt 2> driver_err.txt"
exit /b 0
