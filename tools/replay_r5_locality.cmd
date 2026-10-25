@echo off
rem RemoteSigned applies only to this child process; persistent execution policies are unchanged.
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0replay_r5_locality.ps1" %*
exit /b %errorlevel%
