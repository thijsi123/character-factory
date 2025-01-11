# main.py

import gradio as gr
from ui import create_webui

if __name__ == "__main__":
    webui = create_webui()
    webui.launch(debug=True)