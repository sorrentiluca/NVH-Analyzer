@echo off
rem ============================================================================
rem  Run NVH Analyzer  --  starts the bundled, no-install NVH Analyzer
rem  Double-click this file. A browser tab opens at http://localhost:8501.
rem  Closing this black window stops the app.
rem ============================================================================
setlocal
cd /d "%~dp0"

rem Use the bundled runtime only; never the system Python.
set "PYHOME=%~dp0runtime"
set "PYEXE=%PYHOME%\python.exe"
set "PYTHONPATH=%~dp0app"

rem Keep matplotlib/streamlit caches inside the user profile so the app folder
rem can live on a read-only share or USB stick.
set "MPLCONFIGDIR=%LOCALAPPDATA%\NVH-Analyzer\mpl"
set "STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>&1

if not exist "%PYEXE%" (
  echo [ERROR] Bundled runtime not found at "%PYEXE%".
  echo This folder looks incomplete - re-extract the full zip and try again.
  pause
  exit /b 1
)

rem Open the browser a few seconds after the server has had time to start.
start "" /b cmd /c "timeout /t 6 >nul & start "" http://localhost:8501"

echo Starting NVH Analyzer...  (leave this window open; close it to quit)
"%PYEXE%" -m streamlit run "%~dp0app\streamlit_app.py"

pause
