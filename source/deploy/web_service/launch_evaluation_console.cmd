@echo off
setlocal
set "SERVER=%~1"
if "%SERVER%"=="" set "SERVER=tangjiarui@222.195.72.220"
set "REMOTE_REPOSITORY=%~2"
if "%REMOTE_REPOSITORY%"=="" set "REMOTE_REPOSITORY=/staff/tangjiarui/OPC/spaceqa-agent"
echo SpaceQA evaluation console
echo Server: %SERVER%
echo Remote repository: %REMOTE_REPOSITORY%
echo Keep this window open while using the browser console.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0connect_evaluation.ps1" -Server "%SERVER%" -RemoteRepository "%REMOTE_REPOSITORY%" -StartRemoteService
if errorlevel 1 (
  echo.
  echo Connection failed. Check SSH access and the server address.
  pause
)
