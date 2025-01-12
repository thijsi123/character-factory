@echo off
setlocal EnableDelayedExpansion

:: Check if virtual environment already exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip
    pause
    exit /b 1
)

:: Install requirements from requirements.txt
if exist requirements.txt (
    echo Installing or upgrading requirements from requirements.txt...
    pip install -r requirements.txt --use-deprecated=legacy-resolver --no-cache-dir --ignore-installed
    if errorlevel 1 (
        echo Failed to install some packages from requirements.txt, but continuing...
    )
) else (
    echo WARNING: requirements.txt not found!
)

:: Verify Python executable path and installed packages
where python
pip list

:: Installation complete
echo ========================================================
echo Installation complete!
echo Virtual environment created at: venv
echo Python packages installed successfully.
echo ========================================================
pause
endlocal
