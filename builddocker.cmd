@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM Derive image name from descriptor.json (strip namespace for local build)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-Content '%REPO_ROOT%\descriptor.json' | ConvertFrom-Json).'container-image'.image.Split('/')[-1]"`) do set "IMAGE_NAME=%%I"

REM Read version from version.txt
set /p VERSION=<"%REPO_ROOT%\version.txt"

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to change directory to %REPO_ROOT%
    exit /b 1
)

docker build -t %IMAGE_NAME%:%VERSION% -t %IMAGE_NAME%:latest %* .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
