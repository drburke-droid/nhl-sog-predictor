@echo off
title NHL SOG Predictor
cd /d "%~dp0"
echo Starting NHL SOG Predictor...
echo.
echo Dashboard will be at: http://localhost:5000
echo Press Ctrl+C to stop.
echo.
py -3.10 app.py
pause
