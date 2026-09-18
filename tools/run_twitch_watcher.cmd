@echo off
cd /d "%~dp0.."
if not exist "output" mkdir "output"
".venv\Scripts\python.exe" -u "tools\twitch_watcher_gui.py" >> "output\twitch_watcher.log" 2>&1
