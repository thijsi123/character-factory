@echo off

REM Activating the Conda environment
call conda activate charfact

REM Running the Python script in the activated Conda environment
python "app\oobabooga - webui.py"

REM Check for an error and pause if there is one
if %ERRORLEVEL% neq 0 pause

pause
