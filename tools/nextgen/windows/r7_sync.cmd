@echo off
rem Claude: bring C:\myOpenTPU\openTPU-r7 to MarsWise feature/r7-fpga (at least 8a4a7ac3, pushed),
rem fast-forward only. From GitHub; if that fails, from r7.bundle here. Changes nothing else.
setlocal
set WANT=8a4a7ac3702bd4dcd4b0cc825d7139dc332fe719
cd /d C:\myOpenTPU\openTPU-r7 || (echo no C:\myOpenTPU\openTPU-r7 & pause & exit /b 1)
for /f %%b in ('git symbolic-ref --short HEAD') do set BR=%%b
if not "%BR%"=="feature/r7-fpga" (echo on branch %BR%, not feature/r7-fpga: stop & pause & exit /b 1)
git status --short --untracked-files=no
git fetch https://github.com/MarsWise/openTPU.git refs/heads/feature/r7-fpga
if errorlevel 1 (
  echo GitHub fetch failed: using r7.bundle
  git bundle verify "%~dp0r7.bundle" || (echo bad bundle & pause & exit /b 1)
  git fetch "%~dp0r7.bundle" r7 || (echo bundle fetch failed & pause & exit /b 1)
)
git merge --ff-only FETCH_HEAD || (echo NOT a fast-forward: local changes, stop & pause & exit /b 1)
git log --oneline -12
git merge-base --is-ancestor %WANT% HEAD || (echo HEAD does not contain %WANT% & pause & exit /b 1)
for /f %%h in ('git rev-parse --short=8 HEAD') do set HEADSHA=%%h
echo OK: openTPU-r7 at %HEADSHA% (contains 8a4a7ac3)
pause
