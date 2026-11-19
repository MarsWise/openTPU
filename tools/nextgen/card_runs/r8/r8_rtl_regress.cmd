@echo off
rem Claude: the full test suite (3 600+ tests, the RTL ones on Verilator) of rtl_src.tar.gz's tree (git archive of one commit) in a
rem Linux: WSL when it has verilator and python3 (as it is, no Docker), else a container with Docker
rem Desktop. The regression the cloud sandbox (2 CPUs) cannot finish.
rem   r8_rtl_regress.cmd [workers]       default 8 parallel pytest workers, at most one per 2 GiB the container has
rem   (Docker Desktop: Settings, Resources, Memory; 16 GiB or more for 8 workers)
rem Needs internet for pip (numpy, pytest, pytest-xdist, in a venv); Docker only without WSL verilator.
rem The image: verilator/verilator:v5.046 (the version the tests were run with), else :latest.
rem Results in rtlreg\ beside this script: regress.log, summary.txt, junit.xml. Several hours.
rem No card, no FPGA, no host 192.168.0.157: this machine's CPUs only.
setlocal
set "N=%~1"
if "%N%"=="" set "N=8"
set "W=%~dp0rtlreg"
if not exist "%~dp0rtl_src.tar.gz" (echo no rtl_src.tar.gz beside this script & pause & exit /b 1)
if not exist "%W%" mkdir "%W%"
copy /y "%~dp0rtl_src.tar.gz" "%W%\rtl_src.tar.gz" >nul || (echo cannot copy rtl_src.tar.gz & pause & exit /b 1)
copy /y "%~dp0r8_rtl_regress.sh" "%W%\r8_rtl_regress.sh" >nul || (echo cannot copy r8_rtl_regress.sh & pause & exit /b 1)
rem WSL first: a Linux there with verilator and python3 runs the suite as it is, no Docker
echo === checking WSL for verilator and python3
rem the WSL Ubuntu of install_verilator_5046.cmd: /opt/verilator/bin is not on its PATH
wsl bash -c "PATH=/opt/verilator/bin:$PATH; command -v verilator && command -v python3" >nul 2>&1 || goto docker
set "WW="
for /f "usebackq delims=" %%P in (`wsl wslpath -a "%W%"`) do set "WW=%%P"
if not defined WW goto docker
echo === WSL: %N% workers, results in %W% (the progress also in regress.log)
wsl bash -c "tr -d '\r' < '%WW%/r8_rtl_regress.sh' > /tmp/otpu_regress.sh && PATH=/opt/verilator/bin:$PATH REGRESS_W='%WW%' OTPU_VENV=$HOME/otpu_venv bash /tmp/otpu_regress.sh %N%"
set RC=%errorlevel%
goto done
:docker
echo === no verilator / python3 in WSL: Docker instead
echo === checking Docker (Docker Desktop must be running)
docker version >nul 2>&1 || (echo Docker is not running: start Docker Desktop, then run this again. & pause & exit /b 1)
rem the image: one already on this machine first (no download), v5.046 before latest
set "IMG="
for %%I in (verilator/verilator:v5.046 verilator/verilator:latest) do if not defined IMG docker image inspect %%I >nul 2>&1 && set "IMG=%%I"
if defined IMG echo === using the local image %IMG%
if not defined IMG (
  echo === downloading verilator/verilator:v5.046, a few hundred MB: the progress follows
  docker pull verilator/verilator:v5.046 && set "IMG=verilator/verilator:v5.046"
)
if not defined IMG (
  echo === no v5.046 on Docker Hub: verilator/verilator:latest
  docker pull verilator/verilator:latest && set "IMG=verilator/verilator:latest"
)
if not defined IMG (echo cannot get a Verilator image: check the internet / Docker Hub sign-in & pause & exit /b 1)
echo === image %IMG%, %N% workers; the container's progress is also in %W%\regress.log
docker run --rm -v "%W%:/w" --entrypoint bash %IMG% -c "tr -d '\r' < /w/r8_rtl_regress.sh > /tmp/r.sh && bash /tmp/r.sh %N%"
set RC=%errorlevel%
:done
echo.
echo === exit %RC%; send %W%\summary.txt (and regress.log if anything failed)
if exist "%W%\summary.txt" type "%W%\summary.txt"
pause
exit /b %RC%
