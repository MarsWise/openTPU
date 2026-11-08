@echo off
rem Claude: load the r8 bitstream into the FPGA over JTAG, volatile (no flash). The host's PCIe
rem link drops while the FPGA reconfigures: stop every card run on 192.168.0.157 first, and run
rem r8_restore_pcie.cmd right after this. Then r8_card_tests.cmd.
rem   r8_load.cmd                    the a4138552 build's otpu.bit
rem   r8_load.cmd C:\path\otpu.bit   another file
setlocal
set "BIT=%~1"
if "%BIT%"=="" set "BIT=C:\myOpenTPU\openTPU-r8\build\vivado-r8-m8-133.33-a4138552\otpu.bit"
set "VIV=C:\Xilinx\Vivado\2024.2\bin\vivado.bat"
if not exist "%BIT%" (echo No bitstream at %BIT%: the build writes otpu.bit only when setup and hold pass. & pause & exit /b 1)
if not exist "%VIV%" (echo Vivado not found at %VIV% & pause & exit /b 1)
tasklist /fi "imagename eq vivado.exe" | find /i "vivado.exe" >nul && (echo A Vivado process is running ^(the build?^): wait for it. & pause & exit /b 1)
echo === bitstream %BIT%
certutil -hashfile "%BIT%" SHA256
set /p OK=Load it into the FPGA now (volatile JTAG)? Type yes: 
if /i not "%OK%"=="yes" (echo Not loaded. & pause & exit /b 1)
set "BITF=%BIT:\=/%"
call "%VIV%" -mode batch -nojournal -log "%~dp0r8_load.log" -source "%~dp0r8_load.tcl" -tclargs "%BITF%"
set RC=%errorlevel%
echo === exit %RC% (log %~dp0r8_load.log)
if "%RC%"=="0" echo Next: r8_restore_pcie.cmd, then r8_card_tests.cmd
pause
exit /b %RC%
