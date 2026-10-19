@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if errorlevel 1 exit /b 1

echo BV Analysis Tool - First-time setup
echo.
if exist ".venv\Scripts\python.exe" goto check_environment

echo Looking for 64-bit Python 3.11, 3.12 or 3.13...
for %%V in (3.13 3.12 3.11) do (
    py -%%V -c "import struct, sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 (
        py -%%V -m venv ".venv"
        if errorlevel 1 goto venv_failed
        goto check_environment
    )
)

python -c "import struct, sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv ".venv"
if errorlevel 1 goto venv_failed

:check_environment
".venv\Scripts\python.exe" -c "import struct, sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto environment_invalid
if exist ".venv\.setup-complete" del /q ".venv\.setup-complete"
if exist ".venv\.setup-complete" goto marker_failed

echo Installing required libraries. Internet access is required.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto install_failed
".venv\Scripts\python.exe" -c "import streamlit, pandas, numpy, plotly, openpyxl, scipy, statsmodels"
if errorlevel 1 goto install_failed

>".venv\.setup-complete" echo ready
if errorlevel 1 goto marker_failed
echo.
echo Setup complete. Double-click run.bat to start the tool.
goto success

:python_missing
echo.
echo Python was not found, or its version is unsupported.
echo Install 64-bit Python 3.11, 3.12 or 3.13 from https://www.python.org/downloads/windows/
echo Enable "Add python.exe to PATH", then run setup.bat again.
goto failure

:venv_failed
echo.
echo Could not create the local Python environment.
echo Extract the ZIP first, and use a folder you can write to.
goto failure

:environment_invalid
echo.
echo The .venv environment is incompatible or was moved from another folder.
echo Close the tool, delete only the .venv folder here, and run setup.bat again.
echo Your patient data is stored separately. See README.md for its location.
goto failure

:install_failed
echo.
echo Library installation or validation failed. See the error above.
echo Check your internet connection, then run setup.bat again.
goto failure

:marker_failed
echo.
echo Could not update the setup status in .venv. Check folder permissions.
goto failure

:failure
if /i not "%~1"=="--no-pause" pause
exit /b 1

:success
if /i not "%~1"=="--no-pause" pause
exit /b 0
