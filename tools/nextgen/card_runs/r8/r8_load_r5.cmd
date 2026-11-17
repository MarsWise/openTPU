@echo off
rem Claude: the control for the DDR3 errors: load r5 45b1c1d3 (133.33 MHz; ECC 0 / 0 on this card on 2026-10-31)
rem over JTAG, volatile. Asks "yes" (r8_load.cmd). Then r8_restore_pcie.cmd, r8_ddr_check.cmd and r8_ddr_diag.cmd:
rem clean again = the card and its memory are sound and the r8 builds are at fault. Back to r8: r8_load_newest.cmd.
rem The r8 card tests and models do not run on r5 (r8_card_tests.cmd would refuse its CFG_HASH).
call "%~dp0r8_load.cmd" "C:\myOpenTPU\openTPU.worktrees\pasted-text-processing\build\vivado-r5-m8-133.33-45b1c1d3-tgsc\otpu.bit" %1
