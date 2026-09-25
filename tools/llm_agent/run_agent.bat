@echo off
rem Model comes from arg or AGENT_MODEL (agent.env) - none hardcoded here.
cd /d "%~dp0"
set MODEL=%1
set MODELARG=
if not "%MODEL%"=="" set MODELARG=--model %MODEL%
python driver.py %MODELARG% --show-thoughts --auto-feed --auto-feed-every 15 --steps 400 --delay 2.5
echo.
echo === driver exited (code %ERRORLEVEL%) ===
pause
