#!/bin/bash

echo "========================================"
echo "Character Factory - macOS Installer"
echo "========================================"
echo
echo "⚠️  EXPERIMENTAL SUPPORT ⚠️"
echo "This is experimental and untested on macOS."
echo "Community contributions welcome!"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[WARNING]${NC} Conda is not found in your PATH!"
    echo
    echo "Please install Miniconda from: https://docs.conda.io/projects/miniconda/en/latest/"
    echo
    echo "Or install via Homebrew:"
    echo "  brew install --cask miniconda"
    echo
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

echo -e "${GREEN}[✓]${NC} Conda found!"

# Initialize conda for bash/zsh (if not already done)
if [[ ! -f ~/.zshrc ]] || ! grep -q "conda initialize" ~/.zshrc; then
    if [[ ! -f ~/.bash_profile ]] || ! grep -q "conda initialize" ~/.bash_profile; then
        echo -e "${YELLOW}[INFO]${NC} Initializing conda..."
        conda init
        echo "Please restart your terminal and run this script again."
        exit 0
    fi
fi

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null

# Check if environment exists
if conda env list | grep -q "character-factory"; then
    echo -e "${GREEN}[✓]${NC} Environment already exists!"
else
    echo -e "${YELLOW}[INFO]${NC} Creating character-factory environment..."
    conda create -n character-factory -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Failed to create environment!"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Environment created!"
fi

# Activate environment
echo -e "${YELLOW}[INFO]${NC} Activating environment..."
conda activate character-factory

# Check Python version
if python --version 2>/dev/null | grep -q "3.11"; then
    echo -e "${GREEN}[✓]${NC} Python 3.11 already installed!"
else
    echo -e "${YELLOW}[INFO]${NC} Installing Python 3.11..."
    conda install python=3.11 -y
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Failed to install Python 3.11!"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Python 3.11 installed!"
fi

# Check if requirements are installed (simple check for gradio)
if python -c "import gradio" &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Requirements already installed!"
else
    echo -e "${YELLOW}[INFO]${NC} Installing requirements for Metal (Apple Silicon)..."
    
    # Install ctransformers with Metal support
    CT_METAL=1 pip install ctransformers --no-binary ctransformers
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}[WARNING]${NC} Metal support installation failed, continuing with CPU support..."
    fi
    
    # Install regular requirements
    pip install -r requirements-webui.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Failed to install requirements!"
        exit 1
    fi
    echo -e "${GREEN}[✓]${NC} Requirements installed!"
fi

echo
echo "========================================"
echo -e "${GREEN}Setup Complete! 🎉${NC}"
echo "========================================"
echo
echo "⚠️  Note: macOS support is experimental."
echo "If you encounter issues, please contribute"
echo "feedback to the GitHub repository."
echo
echo "WebUI Applications:"
echo "1) Mistral WebUI"
echo "2) Zephyr WebUI"
echo "3) Power User WebUI"
echo "4) Character Editor"
echo
echo "Command Line Tools:"
echo "5) Mistral CLI"
echo "6) Zephyr CLI"
echo
echo "Other:"
echo "7) Exit"
echo
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "Starting Mistral WebUI..."
        echo "Open http://localhost:7860/ in your browser"
        python ./app/main-mistral-webui.py
        ;;
    2)
        echo "Starting Zephyr WebUI..."
        echo "Open http://localhost:7860/ in your browser"
        python ./app/main-zephyr-webui.py
        ;;
    3)
        echo "Starting Power User WebUI..."
        echo "Open http://localhost:7860/ in your browser"
        python ./app/main-poweruser-webui.py
        ;;
    4)
        echo "Starting Character Editor..."
        echo "Open http://localhost:7860/ in your browser"
        python ./app/character-editor.py
        ;;
    5)
        echo "Starting Mistral CLI..."
        echo "Example: python ./app/main-mistral.py --topic \"fantasy knight\" --name \"Sir Arthur\""
        echo "Opening new terminal for manual usage..."
        open -a Terminal.app . 2>/dev/null || \
        echo "Please open a new terminal and run: conda activate character-factory"
        ;;
    6)
        echo "Starting Zephyr CLI..."
        echo "Example: python ./app/main-zephyr.py --topic \"anime girl\" --gender \"female\""
        echo "Opening new terminal for manual usage..."
        open -a Terminal.app . 2>/dev/null || \
        echo "Please open a new terminal and run: conda activate character-factory"
        ;;
    *)
        echo
        echo "To run manually next time:"
        echo "1. Open terminal"
        echo "2. Navigate to: $(pwd)"
        echo "3. Run: conda activate character-factory"
        echo "4. Run: python ./app/main-mistral-webui.py"
        echo
        ;;
esac