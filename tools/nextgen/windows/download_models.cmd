@echo off
rem Download the r8 models missing on this host into C:\ai-models (see download_models.py).
setlocal
cd /d "%~dp0\..\..\.."
py -3 -m pip install -q -U huggingface_hub || goto :err
py -3 tools\nextgen\models\download_models.py %*
exit /b %errorlevel%
:err
echo pip failed
exit /b 1
