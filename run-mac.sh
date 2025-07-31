#!/bin/bash
# run-mac.sh - Quick launcher for macOS (after initial setup)

echo "========================================"
echo "Character Factory - macOS Launcher"
echo "========================================"
echo
echo "⚠️  EXPERIMENTAL SUPPORT ⚠️"
echo

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate character-factory

if [ $? -ne 0 ]; then
    echo "Error: Please run ./install-mac.sh first!"
    exit 1
fi

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