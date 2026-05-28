@echo off
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM Derive image name from config.yaml (strip namespace for local build)
set "IMAGE_NAME=cellularimagingcf/w_cideconvolve_bl"

REM Read version from version.txt
set "VERSION=v0.0.4"

pushd "%REPO_ROOT%" >nul
if errorlevel 1 (
    echo Failed to change directory to %REPO_ROOT%
    exit /b 1
)

docker build -t %IMAGE_NAME%:%VERSION% -t %IMAGE_NAME%:latest %* .
set "EXITCODE=%ERRORLEVEL%"

popd >nul
endlocal & exit /b %EXITCODE%
