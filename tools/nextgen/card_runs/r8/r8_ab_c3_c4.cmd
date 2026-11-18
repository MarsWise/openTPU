@echo off
rem Claude: A/B on the same card, same day: C3 (6dce3d19) then C4 (508da1c6), each loaded (volatile
rem JTAG), PCIe restored, then r8_speed on the models whose int8 prefill read DRAM wrong on C4
rem (Qwen3.5-0.8B 475 corrected / 53 uncorrectable on channel 1, Qwen3.5-2B 828 / 111, LFM2.5 9 / 1).
rem Logs r8_ab_6dce3d19.log and r8_ab_508da1c6.log here. About 30 minutes. Type yes at each load
rem and the sudo password at each PCIe restore. Leaves C4 loaded; nothing is built.
rem   r8_ab_c3_c4.cmd [models]      default: qwen35 qwen35-2b lfm2 qwen3
setlocal
set "MODELS=%*"
if not defined MODELS set "MODELS=qwen35 qwen35-2b lfm2 qwen3"
set "B=C:\myOpenTPU\openTPU-r8\build"
for %%b in (6dce3d19 508da1c6) do (
  if not exist "%B%\vivado-r8-m8-133.33-%%b\otpu.bit" (echo no %B%\vivado-r8-m8-133.33-%%b\otpu.bit & pause & exit /b 1)
)
for %%b in (6dce3d19 508da1c6) do call :one %%b
echo.
echo === A/B done: r8_ab_6dce3d19.log, r8_ab_508da1c6.log
findstr /c:"########" /c:"after:" /c:"FAILED" /c:"TTFT" "%~dp0r8_ab_6dce3d19.log" "%~dp0r8_ab_508da1c6.log"
pause
exit /b 0
:one
echo.
echo ================ %1: load
call "%~dp0r8_load.cmd" "%B%\vivado-r8-m8-133.33-%1\otpu.bit" || (echo load failed & exit /b 1)
echo ================ %1: PCIe
call "%~dp0r8_restore_pcie.cmd" || (echo PCIe restore failed & exit /b 1)
echo ================ %1: r8_speed %MODELS%
call "%~dp0r8_speed.cmd" %MODELS%
copy /y "%~dp0r8_speed.log" "%~dp0r8_ab_%1.log" >nul
exit /b 0
