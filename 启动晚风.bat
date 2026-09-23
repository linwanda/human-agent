@echo off
cd /d "%~dp0"
title Wanfeng Agent

set "PY="
for /f "delims=" %%i in ('where python 2^>nul') do (
  echo %%i | findstr /i "WindowsApps" >nul
  if errorlevel 1 if not defined PY set "PY=%%i"
)
if not defined PY (
  where py >nul 2>&1
  if not errorlevel 1 set "PY=py"
)

if not defined PY (
  echo [X] Python not found. Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)

if not exist "data\kb.sqlite3" (
  echo [i] First run: creating database...
  if /i "%PY%"=="py" (py -3 cli.py init) else ("%PY%" cli.py init)
  if errorlevel 1 (
    echo [X] init failed.
    pause
    exit /b 1
  )
)

echo.
echo   Wanfeng Agent starting...
echo   If port 8765 is busy, the old Python process will be stopped.
echo   Keep this window open. Close it to stop the server.
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8765 .*LISTENING"') do (
  tasklist /FI "PID eq %%a" | findstr /i "python.exe py.exe" >nul
  if not errorlevel 1 (
    echo [i] Port 8765 in use, stopping PID %%a
    taskkill /F /PID %%a >nul 2>&1
  )
)

start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8765"

if /i "%PY%"=="py" (
  py -3 -u cli.py web --port 8765
) else (
  "%PY%" -u cli.py web --port 8765
)
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
  echo [X] Server exited with code %ERR%
  echo     Copy the error text above if you need help.
)
echo Server stopped.
pause
exit /b %ERR%
