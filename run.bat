@echo off
cd H:\AI\SillyTavern-Launcher-main\SillyTavern-MainBranch
set NODE_ENV=production
call npm install --no-audit --no-fund --quiet --omit=dev
if %errorlevel% neq 0 exit /b %errorlevel% |
node server.js %*
pause