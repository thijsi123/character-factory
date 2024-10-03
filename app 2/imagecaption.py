from PIL import Image
import numpy as np
import onnxruntime as rt
import pandas as pd
import huggingface_hub

# Constants (adjust these paths and model repository names as needed)
HF_TOKEN = "your_hf_token_here"
MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"  # Example model repository
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

def download_model(model_repo, hf_token):
    csv_path = huggingface_hub.hf_hub_download(
        model_repo, LABEL_FILENAME, use_auth_token=hf_token)
    model_path = huggingface_hub.hf_hub_download(
        model_repo, MODEL_FILENAME, use_auth_token=hf_token)
    return csv_path, model_path

def load_model(model_repo, hf_token):
    csv_path, model_path = download_model(model_repo, hf_token)
    tags_df = pd.read_csv(csv_path)
    model = rt.InferenceSession(model_path)
    _, height, width, _ = model.get_inputs()[0].shape
    return model, tags_df, height

def prepare_image(image_array, target_size):
    # Convert NumPy array to PIL Image if not already a PIL Image
    if isinstance(image_array, np.ndarray):
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = image_array  # Assuming it's already a PIL Image

    # Resize and prepare the image as before
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((target_size, target_size), Image.BICUBIC)
    image_array = np.array(pil_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]  # Convert RGB to BGR if needed; depends on model
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(model, image_array):
    input_name = model.get_inputs()[0].name
    preds = model.run(None, {input_name: image_array})[0]
    return preds


def get_sorted_general_strings(image_or_path, model_repo=MODEL_REPO, hf_token=HF_TOKEN):
    # Debug statement
    print(f"Received image_or_path: {image_or_path}")

    # Check if image_or_path is a path (str) or already a PIL Image
    if isinstance(image_or_path, str):
        try:
            pil_image = Image.open(image_or_path)
        except Exception as e:
            print(f"Error opening image: {e}")
            return None  # or handle it in a way that fits your application
    else:
        pil_image = image_or_path  # Assume it's already a PIL Image


