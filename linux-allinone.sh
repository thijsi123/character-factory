#!/bin/bash

# Quick check - if already set up, skip to launcher
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate character-factory &>/dev/null
if [ $? -eq 0 ]; then
    # Check for multiple key modules, not just gradio
    python -c "import gradio, torch, transformers, diffusers" &>/dev/null
    if [ $? -eq 0 ]; then
        launcher
        exit 0
    fi
fi

echo "========================================"
echo "Character Factory - Linux All-in-One"
echo "========================================"
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
    echo "Or install via package manager:"
    echo "  Ubuntu/Debian: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  Arch Linux: yay -S miniconda3"
    echo
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

echo -e "${GREEN}[✓]${NC} Conda found!"

# Initialize conda for bash (if not already done)
if [[ ! -f ~/.bashrc ]] || ! grep -q "conda initialize" ~/.bashrc; then
    echo -e "${YELLOW}[INFO]${NC} Initializing conda..."
    conda init bash
    echo "Please restart your terminal and run this script again."
    exit 0
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

# Check if requirements are installed (check multiple key modules)
if python -c "import gradio, torch, transformers, diffusers" &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Requirements already installed!"
else
    echo -e "${YELLOW}[INFO]${NC} Installing requirements..."
    
    # Detect if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}[INFO]${NC} NVIDIA GPU detected! Installing CUDA requirements..."
        pip install -r requirements-webui-cuda.txt
    else
        echo -e "${YELLOW}[INFO]${NC} No NVIDIA GPU detected. Installing CPU requirements..."
        pip install -r requirements-webui.txt
    fi
    
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

launcher() {
    echo
    echo "========================================"
    echo "Character Factory Launcher"
    echo "========================================"
    echo
    echo "WebUI Applications:"
    echo "1) Mistral WebUI"
    echo "2) Zephyr WebUI"
    echo "3) Power User WebUI"
    echo "4) Character Editor"
    echo
    echo "WebUI Applications (CPU Only):"
    echo "5) Mistral WebUI (CPU)"
    echo "6) Zephyr WebUI (CPU)"
    echo "7) Power User WebUI (CPU)"
    echo "8) Character Editor (CPU)"
    echo
    echo "Command Line Tools:"
    echo "9) Mistral CLI"
    echo "10) Zephyr CLI"
    echo "11) Mistral CLI (CPU)"
    echo "12) Zephyr CLI (CPU)"
    echo
    echo "Maintenance:"
    echo "13) Install/Update Requirements"
    echo "14) Install/Update Requirements (CUDA)"
    echo
    echo "Other:"
    echo "15) Exit"
    echo
    read -p "Enter your choice (1-15): " choice

    case $choice in
        1)
            echo "Starting Mistral WebUI..."
            echo "Please wait."
            python ./app/main-mistral-webui.py
            ;;
        2)
            echo "Starting Zephyr WebUI..."
            echo "Please wait."
            python ./app/main-zephyr-webui.py
            ;;
        3)
            echo "Starting Power User WebUI..."
            echo "Please wait."
            python ./app/main-poweruser-webui.py
            ;;
        4)
            echo "Starting Character Editor..."
            echo "Please wait."
            python ./app/character-editor.py
            ;;
        5)
            echo "Starting Mistral WebUI (CPU Only)..."
            echo "Please wait."
            CUDA_VISIBLE_DEVICES="" FORCE_CPU=1 python ./app/main-mistral-webui.py
            ;;
        6)
            echo "Starting Zephyr WebUI (CPU Only)..."
            echo "Please wait."
            CUDA_VISIBLE_DEVICES="" FORCE_CPU=1 python ./app/main-zephyr-webui.py
            ;;
        7)
            echo "Starting Power User WebUI (CPU Only)..."
            echo "Please wait."
            CUDA_VISIBLE_DEVICES="" FORCE_CPU=1 python ./app/main-poweruser-webui.py
            ;;
        8)
            echo "Starting Character Editor (CPU Only)..."
            echo "Please wait."
            CUDA_VISIBLE_DEVICES="" FORCE_CPU=1 python ./app/character-editor.py
            ;;
        9)
            echo "Starting Mistral CLI..."
            echo "Example: python ./app/main-mistral.py --topic \"fantasy knight\" --name \"Sir Arthur\""
            echo "Opening new terminal for manual usage..."
            gnome-terminal -- bash -c "conda activate character-factory; exec bash" 2>/dev/null || \
            xterm -e "conda activate character-factory; exec bash" 2>/dev/null || \
            echo "Please open a new terminal and run: conda activate character-factory"
            ;;
        10)
            echo "Starting Zephyr CLI..."
            echo "Example: python ./app/main-zephyr.py --topic \"anime girl\" --gender \"female\""
            echo "Opening new terminal for manual usage..."
            gnome-terminal -- bash -c "conda activate character-factory; exec bash" 2>/dev/null || \
            xterm -e "conda activate character-factory; exec bash" 2>/dev/null || \
            echo "Please open a new terminal and run: conda activate character-factory"
            ;;
        11)
            echo "Starting Mistral CLI (CPU Only)..."
            echo "Example: python ./app/main-mistral.py --topic \"fantasy knight\" --name \"Sir Arthur\""
            echo "Opening new terminal for manual usage..."
            gnome-terminal -- bash -c "export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory; exec bash" 2>/dev/null || \
            xterm -e "export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory; exec bash" 2>/dev/null || \
            echo "Please open a new terminal and run: export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory"
            ;;
        12)
            echo "Starting Zephyr CLI (CPU Only)..."
            echo "Example: python ./app/main-zephyr.py --topic \"anime girl\" --gender \"female\""
            echo "Opening new terminal for manual usage..."
            gnome-terminal -- bash -c "export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory; exec bash" 2>/dev/null || \
            xterm -e "export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory; exec bash" 2>/dev/null || \
            echo "Please open a new terminal and run: export CUDA_VISIBLE_DEVICES=''; export FORCE_CPU=1; conda activate character-factory"
            ;;
        13)
            echo "Installing/Updating Requirements (CPU)..."
            pip install -r requirements-webui.txt
            if [ $? -eq 0 ]; then
                echo "✓ Requirements installed successfully!"
            else
                echo "✗ Failed to install requirements!"
            fi
            read -p "Press Enter to continue..."
            launcher
            ;;
        14)
            echo "Installing/Updating Requirements (CUDA)..."
            pip install -r requirements-webui-cuda.txt
            if [ $? -eq 0 ]; then
                echo "✓ CUDA requirements installed successfully!"
            else
                echo "✗ Failed to install CUDA requirements!"
            fi
            read -p "Press Enter to continue..."
            launcher
            ;;
        *)
            echo
            echo "To run manually next time:"
            echo "1. Open terminal"
            echo "2. Navigate to: $(pwd)"
            echo "3. Run: conda activate character-factory"
            echo "4. Run: python ./app/main-mistral-webui.py"
            echo
            echo "For CPU-only mode, set these environment variables first:"
            echo "  export CUDA_VISIBLE_DEVICES=''"
            echo "  export FORCE_CPU=1"
            echo
            ;;
    esac
}

launcher