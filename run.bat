@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" goto setup_required
if not exist ".venv\.setup-complete" goto setup_required

echo BV Analysis Tool
echo Keep this window open while using the tool.
echo If your browser does not open, visit http://127.0.0.1:8501
echo Press Ctrl+C in this window to stop the tool.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
if errorlevel 1 goto start_failed
exit /b 0

:setup_required
echo Run setup.bat first. After setup completes, run this file again.
if /i not "%~1"=="--no-pause" pause
exit /b 1

:start_failed
echo.
echo Could not start the tool. See the error above.
echo Run setup.bat again to repair the environment.
echo If port 8501 is in use, close the other running copy of the tool first.
if /i not "%~1"=="--no-pause" pause
exit /b 1
