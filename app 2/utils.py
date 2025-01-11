# utils.py

import os
import requests
from diffusers import StableDiffusionPipeline
import torch
import sys
llm = None
sd = None
safety_checker_sd = None
global_avatar_prompt = ""
processed_image_path = None
global_url = "http://localhost:5001/api/v1/completions"

folder_path = "models"  # Base directory for models

def load_model(model_name, use_safetensors=False, use_local=False):
    global sd
    # Enable TensorFloat-32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True

    if use_local:
        model_path = os.path.join(folder_path, model_name).replace("\\", "/")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}.")
            return
        print(f"Loading local model from: {model_path}")
        sd = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)
    else:
        print(f"Loading {model_name} from Hugging Face with safetensors={use_safetensors}.")
        sd = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=use_safetensors,
                                                     torch_dtype=torch.float16)

    if torch.cuda.is_available():
        sd.to("cuda")
        print(f"Loaded {model_name} to GPU in half precision (float16).")
    elif torch.backends.mps.is_available():
        sd.to("mps")
        print(f"Loaded {model_name} to Metal (MPS) in half precision (float16).")
    else:
        if sys.platform == "darwin":
            sd.to("cpu", torch.float32)
            print(f"Loaded {model_name} to CPU in full precision (float32) on macOS.")
        else:
            print(f"Loaded {model_name} to CPU.")

def process_url(url):
    global global_url
    global_url = url.rstrip("/") + "/api/v1/completions"  # Append '/v1/completions' to the URL
    return f"URL Set: {global_url}"  # Return the modified URL

def send_message(prompt):
    global global_url
    if not global_url:
        return "Error: URL not set."
    request = {
        'prompt': prompt,
        "max_length": 8192,
        "max_new_tokens": 2048,
        "max_tokens": 2048,
        "max_content_length": 8192,
        'do_sample': True,
        'temperature': 1,
        'typical_p': 1,
        'repetition_penalty': 1.1,
        'guidance_scale': 1,
        'sampler_seed': -1,
        'stop': [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
            r"\n",
            r"\nThijs:",
            "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
        "stopping_strings": [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
            r"\n",
            r"\nThijs:",
            "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
    }

    try:
        response = requests.post(global_url, json=request)
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '')
        # Remove all text after '<|'
        result = result.split('<')[0]
        return result
    except requests.RequestException as e:
        return f"Error sending request: {e}"

def input_none(text):
    user_input = text
    if user_input == "":
        return None
    else:
        return user_input

def combined_avatar_prompt_action(prompt):
    # First, update the avatar prompt
    global global_avatar_prompt
    global_avatar_prompt = prompt
    update_message = "Avatar prompt updated!"  # Or any other message you want to display

    # Then, use the updated avatar prompt
    use_message = f"Using avatar prompt: {global_avatar_prompt}"

    # Return both messages or just one, depending on how you want to display the outcome
    return update_message, use_message

# Load the model
load_model("oof.safetensors", use_safetensors=True, use_local=True)

if sd is not None:
    safety_checker_sd = sd.safety_checker
else:
    safety_checker_sd = None
    print("Warning: Model not loaded. Some features may not be available.")
