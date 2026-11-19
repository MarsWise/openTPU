@echo off
rem Claude: the full test suite (3 600+ tests, the RTL ones on Verilator) of rtl_src.tar.gz's tree (git archive of one commit) in a
rem Linux container with Docker Desktop: the regression the cloud sandbox (2 CPUs) cannot finish.
rem   r8_rtl_regress.cmd [workers]       default 8 parallel pytest workers (about 1 GiB each)
rem Needs Docker Desktop running and internet for the image and pip (numpy, pytest, pytest-xdist).
rem The image: verilator/verilator:v5.046 (the version the tests were run with), else :latest.
rem Results in rtlreg\ beside this script: regress.log, summary.txt, junit.xml. Several hours.
rem No card, no FPGA, no host 192.168.0.157: this machine's CPUs only.
setlocal
set "N=%~1"
if "%N%"=="" set "N=8"
set "W=%~dp0rtlreg"
docker version >nul 2>&1 || (echo Docker is not running: start Docker Desktop, then run this again. & pause & exit /b 1)
if not exist "%~dp0rtl_src.tar.gz" (echo no rtl_src.tar.gz beside this script & pause & exit /b 1)
if not exist "%W%" mkdir "%W%"
copy /y "%~dp0rtl_src.tar.gz" "%W%\rtl_src.tar.gz" >nul || (echo cannot copy rtl_src.tar.gz & pause & exit /b 1)
copy /y "%~dp0r8_rtl_regress.sh" "%W%\r8_rtl_regress.sh" >nul || (echo cannot copy r8_rtl_regress.sh & pause & exit /b 1)
set "IMG=verilator/verilator:v5.046"
docker pull %IMG% >nul 2>&1 || (echo no %IMG%: verilator/verilator:latest instead & set "IMG=verilator/verilator:latest")
echo === image %IMG%, %N% workers, results in %W%
docker run --rm -v "%W%:/w" --entrypoint bash %IMG% -c "tr -d '\r' < /w/r8_rtl_regress.sh > /tmp/r.sh && bash /tmp/r.sh %N%"
set RC=%errorlevel%
echo.
echo === exit %RC%; send %W%\summary.txt (and regress.log if anything failed)
if exist "%W%\summary.txt" type "%W%\summary.txt"
pause
exit /b %RC%
