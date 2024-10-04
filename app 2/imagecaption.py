from PIL import Image
import numpy as np
import onnxruntime as rt
import pandas as pd
import huggingface_hub
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
HF_TOKEN = "your_hf_token_here"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Model repositories
MODEL_REPOS = [
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "SmilingWolf/wd-v1-4-vit-tagger-v2"
]

# Global variables to store the model and tags
model = None
tags_df = None
target_size = None
current_model_repo = None


def download_model(model_repo, hf_token=None):
    logger.info(f"Attempting to download model from {model_repo}")
    try:
        csv_path = huggingface_hub.hf_hub_download(
            model_repo, LABEL_FILENAME, use_auth_token=hf_token)
        logger.info(f"Downloaded CSV from {model_repo}: {csv_path}")
        model_path = huggingface_hub.hf_hub_download(
            model_repo, MODEL_FILENAME, use_auth_token=hf_token)
        logger.info(f"Downloaded ONNX model from {model_repo}: {model_path}")
        return csv_path, model_path
    except Exception as e:
        logger.warning(f"Failed to download model from {model_repo}: {str(e)}")
        return None, None


def load_model(model_repo=None, hf_token=None):
    global model, tags_df, target_size, current_model_repo

    if model is None or model_repo != current_model_repo:
        logger.info(f"Loading model from repository: {model_repo if model_repo else 'Automatic model search'}")
        for repo in MODEL_REPOS if model_repo is None else [model_repo]:
            csv_path, model_path = download_model(repo, hf_token)
            if csv_path and model_path:
                try:
                    tags_df = pd.read_csv(csv_path)
                    model = rt.InferenceSession(model_path)
                    _, target_size, _, _ = model.get_inputs()[0].shape
                    current_model_repo = repo
                    logger.info(f"Successfully loaded model from {repo} with target size: {target_size}")
                    return model, tags_df, target_size
                except Exception as e:
                    logger.warning(f"Failed to load model from {repo}: {str(e)}")

    if model is None:
        logger.error("Failed to load any model from available repositories.")
        raise ValueError("Failed to load any model")

    return model, tags_df, target_size


def prepare_image(image_array, target_size):
    logger.info(f"Preparing image for inference. Target size: {target_size}")
    if isinstance(image_array, np.ndarray):
        logger.info("Received image as NumPy array, converting to PIL image.")
        pil_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    else:
        logger.info("Received image as PIL image.")
        pil_image = image_array

    pil_image = pil_image.convert("RGB")
    logger.info("Converting image to RGB and resizing.")
    pil_image = pil_image.resize((target_size, target_size), Image.BICUBIC)
    image_array = np.array(pil_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]  # BGR format
    image_array = np.expand_dims(image_array, axis=0)
    logger.info(f"Image preparation complete. Shape: {image_array.shape}")
    return image_array


def predict(model, image_array):
    logger.info("Running inference on the prepared image.")
    input_name = model.get_inputs()[0].name
    logger.info(f"Model input name: {input_name}")
    preds = model.run(None, {input_name: image_array})[0]
    logger.info(f"Predictions received. Shape: {preds.shape}")
    return preds


def get_sorted_general_strings(image_or_path, model_repo=None, hf_token=HF_TOKEN, general_threshold=0.35):
    global model, tags_df, target_size

    try:
        model, tags_df, target_size = load_model(model_repo, hf_token)
    except ValueError:
        logger.error("Failed to load any model. Please check your internet connection and model repositories.")
        return None

    logger.info(f"Received image_or_path of type: {type(image_or_path)}")

    if isinstance(image_or_path, str):
        logger.info(f"Attempting to open image from path: {image_or_path}")
        try:
            pil_image = Image.open(image_or_path)
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            return None
    elif isinstance(image_or_path, np.ndarray):
        logger.info("Received image as a NumPy array.")
        pil_image = Image.fromarray(image_or_path.astype('uint8'), 'RGB')
    else:
        logger.info("Assuming image is already a PIL Image object.")
        pil_image = image_or_path

    image_array = prepare_image(pil_image, target_size)
    preds = predict(model, image_array)

    logger.info(f"Tag DataFrame loaded with {len(tags_df)} tags.")
    tag_names = tags_df['name'].tolist()
    general_names = list(zip(tag_names, preds[0].astype(float)))

    logger.info(f"Applying threshold: {general_threshold} to filter predictions.")
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)

    logger.info(f"Filtered {len(general_res)} tags after thresholding.")

    sorted_general_strings = sorted(
        general_res.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_general_strings = [x[0] for x in sorted_general_strings]

    logger.info(f"Sorted and filtered tags: {sorted_general_strings}")
    return ", ".join(sorted_general_strings)
