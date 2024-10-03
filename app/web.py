import os
import requests
from bs4 import BeautifulSoup
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import time
os.environ["DIFFUSERS_DISABLE_SAFETY_CHECKER"] = "1"
sd = None

# Setting up the text generation URL (global_url should be set appropriately)
global_url = "http://127.0.0.1:5000/"  # Replace with your actual URL

def send_message(prompt):
    global global_url
    if not global_url:
        return "Error: URL not set."
    request = {
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1,
        "repetition_penalty": 1.07,
        "top_k": 40,
        "n_predict": 1,
    }
    try:
        response = requests.post(global_url, json=request)
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '')
        return result
    except requests.RequestException as e:
        return f"Error sending request: {e}"

# Image generation setup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.backends.cuda.matmul.allow_tf32 = True

def load_model(model_name, use_safetensors=False, use_local=False):
    global sd
    if use_local:
        model_path = os.path.join("models", model_name).replace("\\", "/")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}.")
            return
        sd = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)
    else:
        sd = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=use_safetensors, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        sd.to("cuda")
    else:
        print(f"Loaded {model_name} to CPU.")

# Load your model (replace with your actual model details)
load_model("oof.safetensors", use_safetensors=True, use_local=True)

def generate_image(prompt):
    if not sd:
        return None
    with torch.no_grad():
        image = sd(prompt).images[0]
    return image

# Infinite scrolling and data extraction function
def scrape_data():
    page = 1
    while True:
        url = f"https://example.com/api/characters?page={page}"  # Replace with your actual API URL
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for item in data['results']:
            yield item
        page += 1
        time.sleep(1)  # Respect server by adding delay between requests

def main():
    for item in scrape_data():
        print("Character Name:", item['name'])
        print("Description:", item['description'])
        # Generate text
        text_prompt = f"Describe the character {item['name']}"
        description = send_message(text_prompt)
        print("Generated Description:", description)
        # Generate image
        image_prompt = f"Generate an image of {item['name']}"
        image = generate_image(image_prompt)
        if image:
            image.show()
            image.save(f"{item['name']}.png")

if __name__ == "__main__":
    main()