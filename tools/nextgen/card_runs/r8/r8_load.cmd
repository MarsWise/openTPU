@echo off
rem Claude: load the r8 bitstream into the FPGA over JTAG, volatile (no flash). The host's PCIe
rem link drops while the FPGA reconfigures: stop every card run on 192.168.0.157 first, and run
rem r8_restore_pcie.cmd right after this. Then r8_card_tests.cmd.
rem   r8_load.cmd                    the 2c800a8f build's otpu.bit
rem   r8_load.cmd C:\path\otpu.bit   another file
setlocal
set "BIT=%~1"
if "%BIT%"=="" set "BIT=C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-2c800a8f\otpu.bit"
set "VIV=C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
if not exist "%BIT%" (echo No bitstream at %BIT%: the build writes otpu.bit only when setup and hold pass. & pause & exit /b 1)
if not exist "%VIV%" (echo Vivado not found at %VIV% & pause & exit /b 1)
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running ^(the build?^): wait for it. & pause & exit /b 1)
echo === bitstream %BIT%
certutil -hashfile "%BIT%" SHA256
set /p OK=Load it into the FPGA now (volatile JTAG)? Type yes: 
if /i not "%OK%"=="yes" (echo Not loaded. & pause & exit /b 1)
rem stale hw_server / cs_server from an earlier session break connect_hw_server (no effect on the FPGA)
taskkill /f /im cs_server.exe >nul 2>&1
taskkill /f /im hw_server.exe >nul 2>&1
set "BITF=%BIT:\=/%"
call "%VIV%" -mode batch -nojournal -log "%~dp0r8_load.log" -source "%~dp0r8_load.tcl" -tclargs "%BITF%"
set RC=%errorlevel%
rem Vivado batch can exit 0 after a Tcl error: success only if the log shows the device programmed with DONE=1
findstr /c:"PROGRAMMED_DEVICE" "%~dp0r8_load.log" | findstr /c:"DONE=1" >nul || set RC=1
echo === exit %RC% (log %~dp0r8_load.log)
if "%RC%"=="0" (echo LOADED. Next: r8_restore_pcie.cmd, then r8_card_tests.cmd) else (echo NOT LOADED: do not run r8_restore_pcie.cmd. Send r8_load.log.)
pause
exit /b %RC%
