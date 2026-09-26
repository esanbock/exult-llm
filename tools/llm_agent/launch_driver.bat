@echo off
rem Fully-detached driver launcher. Starts python in its own window with no
rem inherited stdout/stderr handles, so the caller returns immediately.
rem
rem Model resolution order: 1) command-line arg  2) AGENT_MODEL env / agent.env.
rem NO model name is hardcoded here (deployment-specific config lives in the
rem gitignored agent.env). Usage: launch_driver.bat [model]
cd /d "%~dp0"
set MODEL=%1
set MODELARG=
if not "%MODEL%"=="" set MODELARG=--model %MODEL%
rem Show the turn log LIVE in this console window while ALSO teeing it to
rem driver_log.txt (errors still go to driver_err.txt). The window opens normal
rem (not /min) so you can watch the TURN ... | action | reason | result lines.
start "llm-driver" cmd /c "python -u driver.py %MODELARG% --show-thoughts --auto-feed --auto-feed-every 15 --save-every 40 --num-ctx 32768 --steps 20000 --delay 2.0 2> driver_err.txt | powershell -NoProfile -Command \"$input | Tee-Object -FilePath driver_log.txt\""
exit /b 0
