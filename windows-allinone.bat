@echo off
setlocal enabledelayedexpansion

REM Quick check - if already set up, skip to launcher
call conda activate character-factory >nul 2>&1
if %errorlevel% equ 0 (
    REM Check for multiple key modules, not just gradio
    python -c "import gradio, torch, transformers, diffusers" >nul 2>&1
    if %errorlevel% equ 0 (
        goto :launcher
    )
)

echo ========================================
echo Character Factory - Windows All-in-One
echo ========================================
echo.

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Conda is not found in your PATH!
    echo Please install Miniconda and make sure it's added to PATH.
    pause
    exit /b 1
)

echo [✓] Conda found!

REM Check if environment exists
conda env list | findstr "character-factory" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Creating character-factory environment...
    conda create -n character-factory -y
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create environment!
        pause
        exit /b 1
    )
    echo [✓] Environment created!
) else (
    echo [✓] Environment already exists!
)

REM Activate the environment
echo [INFO] Activating environment...
call conda activate character-factory
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate environment!
    echo [INFO] You may need to restart your command prompt after conda installation.
    pause
    exit /b 1
)

REM Check Python version
python --version 2>nul | findstr "3.11" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing Python 3.11...
    conda install python=3.11 -y
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install Python 3.11!
        pause
        exit /b 1
    )
    echo [✓] Python 3.11 installed!
) else (
    echo [✓] Python 3.11 already installed!
)

REM Check if requirements are installed (check multiple key modules)
python -c "import gradio, torch, transformers, diffusers" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing requirements...
    
    REM Detect if NVIDIA GPU is available
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] NVIDIA GPU detected! Installing CUDA requirements...
        pip install -r requirements-webui-cuda.txt
    ) else (
        echo [INFO] No NVIDIA GPU detected. Installing CPU requirements...
        pip install -r requirements-webui.txt
    )
    
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install requirements!
        pause
        exit /b 1
    )
    echo [✓] Requirements installed!
) else (
    echo [✓] Requirements already installed!
)

echo.
echo ========================================
echo Setup Complete! 🎉
echo ========================================

:launcher
echo.
echo ========================================
echo Character Factory Launcher
echo ========================================
echo.
echo WebUI Applications:
echo 1) Mistral WebUI
echo 2) Zephyr WebUI  
echo 3) Power User WebUI
echo 4) Character Editor
echo.
echo WebUI Applications (CPU Only):
echo 5) Mistral WebUI (CPU)
echo 6) Zephyr WebUI (CPU)
echo 7) Power User WebUI (CPU)
echo 8) Character Editor (CPU)
echo.
echo Command Line Tools:
echo 9) Mistral CLI
echo 10) Zephyr CLI
echo 11) Mistral CLI (CPU)
echo 12) Zephyr CLI (CPU)
echo.
echo Maintenance:
echo 13) Install/Update Requirements
echo 14) Install/Update Requirements (CUDA)
echo.
echo Other:
echo 15) Exit
echo.
set /p choice="Enter your choice (1-15): "

if "%choice%"=="1" goto :option1
if "%choice%"=="2" goto :option2
if "%choice%"=="3" goto :option3
if "%choice%"=="4" goto :option4
if "%choice%"=="5" goto :option5
if "%choice%"=="6" goto :option6
if "%choice%"=="7" goto :option7
if "%choice%"=="8" goto :option8
if "%choice%"=="9" goto :option9
if "%choice%"=="10" goto :option10
if "%choice%"=="11" goto :option11
if "%choice%"=="12" goto :option12
if "%choice%"=="13" goto :install_req
if "%choice%"=="14" goto :install_req_cuda
if "%choice%"=="15" goto :exit
goto :invalid_choice

:option1
echo Starting Mistral WebUI...
echo Please wait.
python ./app/main-mistral-webui.py
goto :end

:option2
echo Starting Zephyr WebUI...
echo Please wait.
python ./app/main-zephyr-webui.py
goto :end

:option3
echo Starting Power User WebUI...
echo Please wait.
python ./app/main-poweruser-webui.py
goto :end

:option4
echo Starting Character Editor...
echo Please wait.
python ./app/character-editor.py
goto :end

:option5
echo Starting Mistral WebUI (CPU Only)...
echo Please wait.
set CUDA_VISIBLE_DEVICES=
set FORCE_CPU=1
python ./app/main-mistral-webui.py
goto :end

:option6
echo Starting Zephyr WebUI (CPU Only)...
echo Please wait.
set CUDA_VISIBLE_DEVICES=
set FORCE_CPU=1
python ./app/main-zephyr-webui.py
goto :end

:option7
echo Starting Power User WebUI (CPU Only)...
echo Please wait.
set CUDA_VISIBLE_DEVICES=
set FORCE_CPU=1
python ./app/main-poweruser-webui.py
goto :end

:option8
echo Starting Character Editor (CPU Only)...
echo Please wait.
set CUDA_VISIBLE_DEVICES=
set FORCE_CPU=1
python ./app/character-editor.py
goto :end

:option9
echo Starting Mistral CLI...
echo Example: python ./app/main-mistral.py --topic "fantasy knight" --name "Sir Arthur"
echo Press any key to open command prompt for manual usage...
pause >nul
cmd /k "conda activate character-factory"
goto :end

:option10
echo Starting Zephyr CLI...
echo Example: python ./app/main-zephyr.py --topic "anime girl" --gender "female"
echo Press any key to open command prompt for manual usage...
pause >nul
cmd /k "conda activate character-factory"
goto :end

:option11
echo Starting Mistral CLI (CPU Only)...
echo Example: python ./app/main-mistral.py --topic "fantasy knight" --name "Sir Arthur"
echo Press any key to open command prompt for manual usage...
pause >nul
cmd /k "set CUDA_VISIBLE_DEVICES= && set FORCE_CPU=1 && conda activate character-factory"
goto :end

:option12
echo Starting Zephyr CLI (CPU Only)...
echo Example: python ./app/main-zephyr.py --topic "anime girl" --gender "female"
echo Press any key to open command prompt for manual usage...
pause >nul
cmd /k "set CUDA_VISIBLE_DEVICES= && set FORCE_CPU=1 && conda activate character-factory"
goto :end

:install_req
echo Installing/Updating Requirements (CPU)...
pip install -r requirements-webui.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements!
) else (
    echo [✓] Requirements installed successfully!
)
pause
goto :launcher

:install_req_cuda
echo Installing/Updating Requirements (CUDA)...
pip install -r requirements-webui-cuda.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install CUDA requirements!
) else (
    echo [✓] CUDA requirements installed successfully!
)
pause
goto :launcher

:invalid_choice
echo Invalid choice! Please enter a number between 1-15.
pause
goto :launcher

:exit
echo.
echo To run manually next time:
echo 1. Open Command Prompt or Anaconda Prompt
echo 2. Navigate to: %~dp0
echo 3. Run: conda activate character-factory
echo 4. Run: python ./app/main-mistral-webui.py
echo.
echo For CPU-only mode, set these environment variables first:
echo   set CUDA_VISIBLE_DEVICES=
echo   set FORCE_CPU=1
echo.

:end
pause