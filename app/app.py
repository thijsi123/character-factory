###############################################################################
# 1. IMPORTS & GLOBALS
###############################################################################
import os
import re
import sys
import random
import logging
import requests
import numpy as np
import pandas as pd
import torch
import gradio as gr
import nltk
import json
import time

from PIL import Image, PngImagePlugin
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import StableDiffusionXLPipeline
import onnxruntime as rt
import huggingface_hub

# If you have 'aichar' installed. Otherwise, remove or replace references.
try:
    import aichar
    HAVE_AICHAR = True
except ImportError:
    HAVE_AICHAR = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default global URL for LLM completions
global_url = "http://localhost:5001/api/v1/completions"

# Hugging Face token for imagecaption (change or remove if needed)
HF_TOKEN = "your_hf_token_here"

# Global variables for the UI flow
global_avatar_prompt = ""
processed_image_path = None

# Path to your local SDXL single-file model (e.g., a custom .safetensors)
LOCAL_MODEL_PATH = r"G:\Documents\GitHub\character-factory\models\illustriousXLPersonalMerge_v30Noob10based.safetensors"

# Our SDXL pipeline, loaded below
sd = None

###############################################################################
# 2. IMPORT PROMPT MODULES DYNAMICALLY
###############################################################################
# We'll assume each prompt_*.py has a variable named `example_dialogue`
PROMPTS_DIR = r"G:\Documents\GitHub\character-factory\app\prompts"
PROMPTS_DIR = os.path.abspath(PROMPTS_DIR)

if PROMPTS_DIR not in sys.path:
    sys.path.append(PROMPTS_DIR)

import prompt_name
import prompt_summary
import prompt_personality
import prompt_scenario
import prompt_greeting
import prompt_example
# ... if you have other prompt modules (e.g., prompt_nonnsfw_summary), import them similarly

def safe_get_dialogues(module, fallback):
    """
    Get the 'example_dialogue' list from a module or return a fallback.
    """
    return getattr(module, "example_dialogue", fallback)

PROMPT_NAME_EXAMPLES = safe_get_dialogues(prompt_name, ["fallback name1", "fallback name2"])
PROMPT_SUMMARY_EXAMPLES = safe_get_dialogues(prompt_summary, ["fallback summary1", "fallback summary2"])
PROMPT_PERSONALITY_EXAMPLES = safe_get_dialogues(prompt_personality, ["fallback personality1", "fallback personality2"])
PROMPT_SCENARIO_EXAMPLES = safe_get_dialogues(prompt_scenario, ["fallback scenario1", "fallback scenario2"])
PROMPT_GREETING_EXAMPLES = safe_get_dialogues(prompt_greeting, ["fallback greeting1", "fallback greeting2"])
PROMPT_EXAMPLE_DIALOGUES = safe_get_dialogues(prompt_example, ["fallback example1", "fallback example2"])

###############################################################################
# 3. UTILS / SHARED
###############################################################################

def process_url(url: str):
    global global_url
    global_url = url.rstrip("/") + "/api/v1/completions"
    return f"URL Set: {global_url}"

def send_message(prompt: str) -> str:
    global global_url
    if not global_url:
        return "Error: URL not set."

    request_body = {
        "prompt": prompt,
        "max_length": 8192,
        "max_new_tokens": 2048,
        "max_tokens": 2048,
        "max_content_length": 8192,
        "do_sample": True,
        "temperature": 1,
        "typical_p": 1,
        "repetition_penalty": 1.1,
        "guidance_scale": 1,
        "sampler_seed": -1,
        "stop": [
            "/s","</s>","<s>","<|system|>","<|assistant|>","<|user|>","<|char|>",
            r"\n","\nThijs:","<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n","<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
        "stopping_strings": [
            "/s","</s>","<s>","<|system|>","<|assistant|>","<|user|>","<|char|>",
            r"\n","\nThijs:","<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n","<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
    }
    try:
        resp = requests.post(global_url, json=request_body)
        resp.raise_for_status()
        result = resp.json().get("choices", [{}])[0].get("text", "")
        result = result.split("<")[0]
        return result
    except requests.RequestException as e:
        return f"Error sending request: {e}"

def input_none(text):
    return None if (text == "") else text

def combined_avatar_prompt_action(prompt: str):
    global global_avatar_prompt
    global_avatar_prompt = prompt
    update_message = "Avatar prompt updated!"
    use_message = f"Using avatar prompt: {global_avatar_prompt}"
    return update_message, use_message

###############################################################################
# 4. LOADING SDXL FROM A SINGLE-FILE MODEL
###############################################################################

def load_sdxl_model():
    global sd
    model_path = LOCAL_MODEL_PATH

    if not os.path.exists(model_path):
        logger.error(f"Local model path does not exist: {model_path}")
        sys.exit(1)

    print(f"Loading SDXL model from single-file checkpoint: {model_path}")
    try:
        if torch.cuda.is_available():
            sd = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16
            )
            sd.to("cuda")
            print("SDXL model loaded in float16 on GPU.")
        else:
            sd = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32
            )
            sd.to("cpu")
            print("SDXL model loaded in float32 on CPU.")
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        sys.exit(1)

###############################################################################
# 5. IMAGE CAPTIONING (ONNX)
###############################################################################
import onnxruntime as rt
import huggingface_hub
from PIL import Image

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
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

imagecaption_model = None
tags_df = None
target_size = None
current_model_repo = None

def download_model(model_repo, hf_token=None):
    logger.info(f"Attempting to download model from {model_repo}")
    try:
        csv_path = huggingface_hub.hf_hub_download(
            model_repo, LABEL_FILENAME, use_auth_token=hf_token
        )
        logger.info(f"Downloaded CSV from {model_repo}: {csv_path}")
        model_path = huggingface_hub.hf_hub_download(
            model_repo, MODEL_FILENAME, use_auth_token=hf_token
        )
        logger.info(f"Downloaded ONNX model from {model_repo}: {model_path}")
        return csv_path, model_path
    except Exception as e:
        logger.warning(f"Failed to download model from {model_repo}: {str(e)}")
        return None, None

def load_imagecaption_model(model_repo=None, hf_token=None):
    global imagecaption_model, tags_df, target_size, current_model_repo

    if imagecaption_model is None or model_repo != current_model_repo:
        logger.info(f"Loading imagecaption model from: {model_repo if model_repo else 'Auto-search'}")
        for repo in (MODEL_REPOS if model_repo is None else [model_repo]):
            csv_path, model_path = download_model(repo, hf_token)
            if csv_path and model_path:
                try:
                    tags_df = pd.read_csv(csv_path)
                    imagecaption_model = rt.InferenceSession(model_path)
                    _, target_size_, _, _ = imagecaption_model.get_inputs()[0].shape
                    target_size = target_size_
                    current_model_repo = repo
                    logger.info(f"Loaded imagecaption model from {repo} with target size: {target_size}")
                    return imagecaption_model, tags_df, target_size
                except Exception as e:
                    logger.warning(f"Failed to load model from {repo}: {str(e)}")

    if imagecaption_model is None:
        logger.error("Failed to load any imagecaption model.")
        raise ValueError("Failed to load any model")

    return imagecaption_model, tags_df, target_size

def prepare_image(image_array, target_size_):
    logger.info(f"Preparing image for inference. Target size: {target_size_}")
    if isinstance(image_array, np.ndarray):
        logger.info("Received image as NumPy array, converting to PIL image.")
        pil_image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    else:
        logger.info("Assuming image is already a PIL image.")
        pil_image = image_array

    pil_image = pil_image.convert("RGB")
    logger.info("Converting image to RGB and resizing.")
    pil_image = pil_image.resize((target_size_, target_size_), Image.BICUBIC)
    img_array = np.array(pil_image, dtype=np.float32)
    img_array = img_array[:, :, ::-1]  # Convert to BGR
    img_array = np.expand_dims(img_array, axis=0)
    logger.info(f"Image preparation complete. Shape: {img_array.shape}")
    return img_array

def predict_image_tags(model_session, img_array):
    logger.info("Running inference on the prepared image.")
    input_name = model_session.get_inputs()[0].name
    preds = model_session.run(None, {input_name: img_array})[0]
    logger.info(f"Predictions received. Shape: {preds.shape}")
    return preds

def get_sorted_general_strings(image_or_path, model_repo=None, hf_token=HF_TOKEN, general_threshold=0.35):
    global imagecaption_model, tags_df, target_size

    try:
        imagecaption_model, tags_df, target_size = load_imagecaption_model(model_repo, hf_token)
    except ValueError:
        logger.error("Failed to load any model. Please check your HF token or model repos.")
        return None

    if isinstance(image_or_path, str):
        logger.info(f"Opening image from path: {image_or_path}")
        try:
            pil_image = Image.open(image_or_path)
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            return None
    elif isinstance(image_or_path, np.ndarray):
        pil_image = Image.fromarray(image_or_path.astype('uint8'), 'RGB')
    else:
        pil_image = image_or_path

    img_array = prepare_image(pil_image, target_size)
    preds = predict_image_tags(imagecaption_model, img_array)

    tag_names = tags_df['name'].tolist()
    general_names = list(zip(tag_names, preds[0].astype(float)))

    # Filter by threshold
    filtered = [x for x in general_names if x[1] > general_threshold]
    filtered = dict(filtered)  # convert to dict for sorting

    sorted_tags = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    sorted_tags = [t[0] for t in sorted_tags]

    return ", ".join(sorted_tags)

###############################################################################
# 6. IMPORT/EXPORT
###############################################################################

def find_image_path():
    possible_paths = ["./app2/image.png","./image.png"]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

image_path = find_image_path()
if image_path:
    print(f"Image found at: {image_path}")
else:
    print("Image not found in any of the specified locations.")

def import_character_json(json_path):
    if not HAVE_AICHAR:
        raise ValueError("aichar not installed. Cannot import character JSON.")
    if json_path:
        character = aichar.load_character_json_file(json_path)
        if character.name:
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
    raise ValueError("Error importing from JSON. Check file correctness.")

def import_character_card(card_path):
    if not HAVE_AICHAR:
        raise ValueError("aichar not installed. Cannot import from card.")
    if card_path:
        character = aichar.load_character_card_file(card_path)
        if character.name:
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
    raise ValueError("Error importing from card file. Check file correctness.")

def export_as_json(name, summary, personality, scenario, greeting_message, example_messages, alternate_greetings):
    """
    Build a custom JSON structure manually.
    - Removes "char_greeting" from the final JSON so it doesn't conflict with first_mes
    - Splits user input for alternate greetings by newlines to form an array
    """
    # If user typed multiple lines in 'alternateGreetings', each line => item
    greetings_list = []
    if isinstance(alternate_greetings, str):
        lines = [x.strip() for x in alternate_greetings.split('\n') if x.strip()]
        if lines:
            greetings_list = lines
    elif isinstance(alternate_greetings, list):
        greetings_list = alternate_greetings
    else:
        greetings_list = []

    ms_timestamp = int(time.time() * 1000)

    custom_dict = {
        "chara": {
            "char_name": name,
            "char_persona": personality,
            "world_scenario": scenario,
            # "char_greeting": greeting_message,  # REMOVED
            "example_dialogue": ""
        },
        "name": name,
        "description": summary,
        "personality": personality,
        "scenario": scenario,
        "first_mes": greeting_message,
        "mes_example": example_messages,
        "alternate_greetings": greetings_list,
        "metadata": {
            "version": 1,
            "created": ms_timestamp,
            "modified": ms_timestamp,
            "source": None,
            "tool": {
                "name": "aichar Python library",
                "version": "0.5.1",
                "url": "https://github.com/Hukasx0/aichar"
            }
        }
    }

    return json.dumps(custom_dict, ensure_ascii=False, indent=2)

def export_character_card(name, summary, personality, scenario, greeting_message, example_messages):
    """
    Exports a .card.png using aichar, but it won't embed 'alternate_greetings' or custom fields.
    """
    if not HAVE_AICHAR:
        logger.error("aichar not installed. Cannot export card.")
        return None

    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"
    os.makedirs(base_path, exist_ok=True)

    if processed_image_path and os.path.exists(processed_image_path):
        image_path_ = processed_image_path
    else:
        image_path_ = "characters/uploaded_character/uploaded_character.png"

    if not os.path.exists(image_path_):
        logger.error(f"Image file not found: {image_path_}")
        # Create a default image
        img = Image.new("RGB", (256,256), color=(73,109,137))
        image_path_ = f"{base_path}{character_name}.png"
        img.save(image_path_)

    try:
        character = aichar.create_character(
            name=name,
            summary=summary,
            personality=personality,
            scenario=scenario,
            greeting_message=greeting_message,
            example_messages=example_messages,
            image_path=image_path_
        )
        card_path = f"{base_path}{character_name}.card.png"
        character.export_neutral_card_file(card_path)
        if os.path.exists(card_path):
            return card_path
        else:
            logger.error(f"Character card file not created: {card_path}")
            return None
    except Exception as e:
        logger.exception(f"Error creating character card: {str(e)}")
        return None

def export_custom_png_with_json(
    name, summary, personality, scenario,
    greeting_message, example_messages, alternate_greetings
):
    """
    1) Build your full JSON with export_as_json
    2) If processed_image_path doesn't exist, use fallback
    3) Embed that JSON into a new PNG with the "chara" chunk
    4) Return the path to the newly saved .png
    """
    custom_json_str = export_as_json(
        name, summary, personality, scenario,
        greeting_message, example_messages, alternate_greetings
    )

    global processed_image_path
    if processed_image_path and os.path.exists(processed_image_path):
        source_img_path = processed_image_path
    else:
        source_img_path = "characters/uploaded_character/uploaded_character.png"

    if not os.path.exists(source_img_path):
        logger.warning(f"Fallback: no user-uploaded or generated image found. Creating blank image.")
        # Create a blank fallback
        os.makedirs("characters/uploaded_character", exist_ok=True)
        blank_path = "characters/uploaded_character/fallback.png"
        Image.new("RGB", (256,256), color=(73,109,137)).save(blank_path)
        source_img_path = blank_path

    # Name the output path
    character_name = name.replace(" ", "_") or "CustomChar"
    out_dir = os.path.join("characters", character_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{character_name}.custom.card.png")

    # Read the source image, embed JSON in PNG chunk
    im = Image.open(source_img_path).convert("RGB")
    png_info = PngImagePlugin.PngInfo()
    # We'll store the entire JSON as raw text under the "chara" chunk
    png_info.add_text("chara", custom_json_str)

    im.save(out_path, "PNG", pnginfo=png_info)
    logger.info(f"Exported custom PNG with full JSON to {out_path}")
    return out_path

def process_uploaded_image(uploaded_img, save_as="uploaded.png", out_dir="characters/uploaded_character"):
    global processed_image_path
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(uploaded_img, np.ndarray):
        pil_image = Image.fromarray(np.uint8(uploaded_img)).convert("RGB")
    else:
        pil_image = uploaded_img

    image_path = os.path.join(out_dir, save_as)
    pil_image.save(image_path)
    processed_image_path = image_path
    logger.info(f"Uploaded image saved at: {processed_image_path}")
    return pil_image

###############################################################################
# 7. CHARACTER GENERATION
###############################################################################

def clean_output_brackets(raw_output: str, character_name: str) -> str:
    def ensure_double_brackets(match):
        return "{{" + match.group(1) + "}}"
    cleaned = re.sub(r"\{{3,}(char|user)\}{3,}", ensure_double_brackets, raw_output)
    cleaned = re.sub(r"\{{1,2}(char|user)\}{1,2}", ensure_double_brackets, cleaned)

    if character_name:
        cleaned = re.sub(r"\b" + re.escape(character_name) + r"\b", "{{char}}", cleaned)
    cleaned = re.sub(r'\*\s*"', '"', cleaned)
    cleaned = re.sub(r'"\s*\*', '"', cleaned)
    return cleaned

def generate_character_name(topic, gender, name, surname_checkbox):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_NAME_EXAMPLES)
    gender_txt = f"Character gender: {gender}." if gender else ""
    surname_txt = "Add Surname" if surname_checkbox else ""
    prompt = (
        example_dialogue
        + "\n<|user|> Generate a random character first name. "
        f"Topic: {topic}. {gender_txt} {surname_txt}\n</s>\n<|assistant|>"
    )
    output = send_message(prompt)
    output = re.sub(r"[^a-zA-Z0-9_ -]", "", output).strip()
    return output

def generate_character_summary(character_name, topic, gender):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_SUMMARY_EXAMPLES)
    global global_avatar_prompt
    appearance = global_avatar_prompt

    user_prompt = (
        f"{example_dialogue}\n"
        f"<|user|> Create a longer description for a character named: {character_name}, "
        f"Topic: {topic}, "
        f"{'Character gender: ' + gender + '.' if gender else ''} "
    )
    if appearance:
        user_prompt += f"This character has an appearance of {appearance}. Use those tags. "
    user_prompt += "Make the summary descriptive. </s>\n<|assistant|>"

    output = send_message(user_prompt).strip()
    return output

def generate_character_personality(character_name, character_summary, topic):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_PERSONALITY_EXAMPLES)
    user_prompt = (
        f"{example_dialogue}\n"
        f"<|user|> Describe the personality of {character_name}. "
        f"Their characteristics: {character_summary}. Tailor them to {topic}.\n</s>\n<|assistant|>"
    )
    return send_message(user_prompt).strip()

def generate_character_scenario(character_summary, character_personality, topic):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_SCENARIO_EXAMPLES)
    user_prompt = (
        f"{example_dialogue}\n"
        f"<|user|> Write a scenario for a roleplay with {{char}} & {{user}}. "
        f"{{char}} has these traits: {character_summary}, {character_personality}. "
        f"Theme: {topic}. No dialogues.\n</s>\n<|assistant|>"
    )
    return send_message(user_prompt).strip()

def generate_character_greeting_message(character_name, character_summary, character_personality, topic):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_GREETING_EXAMPLES)
    user_prompt = (
        example_dialogue
        + "\n<|user|> Create the first message from "
        + f"{character_name} who has {character_summary},{character_personality}, "
        + f"theme {topic}, greeting {{user}}.\n</s>\n<|assistant|>"
    )
    raw_output = send_message(user_prompt).strip()
    return clean_output_brackets(raw_output, character_name)

def generate_alternate_greetings(character_name, character_summary, character_personality, topic):
    """
    Generates alternate greetings using the same logic as the main greeting generator.
    """
    example_dialogue = random.choice(PROMPT_GREETING_EXAMPLES)
    user_prompt = (
            example_dialogue
            + "\n<|user|> Create the first message from "
            + f"{character_name} who has {character_summary},{character_personality}, "
            + f"theme {topic}, greeting {{user}}.\n</s>\n<|assistant|>"
    )
    raw_output = send_message(user_prompt).strip()

    # Clean and split output into separate lines for each greeting
    greetings = raw_output.split("\n")
    greetings = [clean_output_brackets(greet.strip(), character_name) for greet in greetings if greet.strip()]
    return greetings


def generate_example_messages(character_name, character_summary, character_personality, topic, switch_function_checkbox):
    # Use dynamically imported examples
    example_dialogue = random.choice(PROMPT_EXAMPLE_DIALOGUES)
    user_prompt = (
        f"{example_dialogue}\n"
        f"<|user|> Create a dialogue between {{user}} and {{char}}. "
        f"{{char}} is {character_name} with traits {character_summary},{character_personality}, "
        f"theme: {topic}. Make it interesting.\n</s>\n<|assistant|>"
    )
    raw_output = send_message(example_dialogue + user_prompt).strip()
    return clean_output_brackets(raw_output, character_name)

###############################################################################
# 8. IMAGE GENERATION
###############################################################################

def fetch_avatar_prompt(character_summary, topic, gender):
    example_dialogue = random.choice(PROMPT_SUMMARY_EXAMPLES)  # or any module as needed
    system_prompt = f"""
<|system|>
You are a text generation tool, returning only tags describing the character's appearance.
</s>
<|user|>: 
create a prompt that lists the appearance characteristics of a character 
whose gender is {gender}, 
whose summary is: {character_summary}.
Theme: {topic}
</s>
<|assistant|>:
"""
    raw_prompt = send_message(system_prompt)
    return raw_prompt.strip()

def generate_character_avatar(character_name, character_summary, topic, negative_prompt, avatar_prompt, gender):
    if not avatar_prompt.strip():
        avatar_prompt = fetch_avatar_prompt(character_summary, topic, gender)

    style_prefix = "anime, 2d, " if ("anime" in character_summary.lower() or "anime" in topic.lower()) else "realistic, 3d, "
    final_prompt = style_prefix + avatar_prompt.strip()

    return image_generate(character_name, final_prompt, input_none(negative_prompt))

def image_generate(character_name, prompt, negative_prompt):
    global processed_image_path
    default_negative = (
        "worst quality, low quality, blurry, text, watermark, logo, banner, "
        "duplicate, poorly drawn, bad anatomy, missing limbs, extra limbs, "
        "ugly face, multiple faces, etc."
    )
    negative_prompt = default_negative + " " + (negative_prompt or "")

    try:
        result = sd(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=1152,
            width=768,
            num_inference_steps=50,
            guidance_scale=5,
        )
        generated_image = result.images[0]
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

    safe_name = character_name.replace(" ", "_")
    out_dir = os.path.join("characters", safe_name)
    os.makedirs(out_dir, exist_ok=True)
    image_path = os.path.join(out_dir, f"{safe_name}.png")
    generated_image.save(image_path, format="PNG")

    np_image = np.array(generated_image)
    process_uploaded_image(np_image, save_as=f"{safe_name}_uploaded.png", out_dir=out_dir)
    return generated_image

###############################################################################
# 9. INFINITE ALTERNATE GREETING LINES
###############################################################################

# This section is already handled within the Gradio UI as Alternate Greetings
# via the `alternateGreetings` textarea and the `generate_alternate_greetings` function.

###############################################################################
# 10. UI / MAIN
###############################################################################

def generate_tags_and_set_prompt(image):
    tags = get_sorted_general_strings(image)
    return tags

def create_webui():
    with gr.Blocks() as webui:
        # State for alternate greetings is managed via the textarea

        gr.Markdown("# Character Factory WebUI (SDXL Single-file)")
        gr.Markdown("## KOBOLD MODE (No Wiki), with dynamic prompt modules")

        with gr.Row():
            url_input = gr.Textbox(label="Enter URL", value="http://127.0.0.1:5001")
            submit_button = gr.Button("Set URL")
        url_status = gr.Textbox(label="URL Status", interactive=False)
        submit_button.click(process_url, inputs=url_input, outputs=url_status)

        with gr.Tab("Edit character"):
            topic = gr.Textbox(label="Topic", placeholder="e.g. Fantasy, Anime, etc.")
            gender = gr.Textbox(label="Gender", placeholder="M, F, or similar")

            with gr.Column():
                # Character name
                with gr.Row():
                    name = gr.Textbox(label="Name", placeholder="Character name")
                    surname_checkbox = gr.Checkbox(label="Add Surname", value=False)
                    name_button = gr.Button("Generate character name with LLM")
                    name_button.click(
                        generate_character_name,
                        inputs=[topic, gender, name, surname_checkbox],
                        outputs=name
                    )

                # Summary
                with gr.Row():
                    summary = gr.Textbox(label="Summary", placeholder="Character summary")
                    summary_button = gr.Button("Generate character summary with LLM (topic included)")
                    summary_button.click(
                        generate_character_summary,
                        inputs=[name, topic, gender],
                        outputs=summary
                    )

                # Personality
                with gr.Row():
                    personality = gr.Textbox(label="Personality", placeholder="Character personality")
                    personality_button = gr.Button("Generate character personality with LLM")
                    personality_button.click(
                        generate_character_personality,
                        inputs=[name, summary, topic],
                        outputs=personality
                    )

                # Scenario
                with gr.Row():
                    scenario = gr.Textbox(label="Scenario", placeholder="Character scenario")
                    scenario_button = gr.Button("Generate character scenario with LLM")
                    scenario_button.click(
                        generate_character_scenario,
                        inputs=[summary, personality, topic],
                        outputs=scenario
                    )

                # Greeting message
                with gr.Row():
                    greeting_message = gr.Textbox(label="Greeting Message", placeholder="Character greeting message")
                    greeting_message_button = gr.Button("Generate character greeting message with LLM")
                    greeting_message_button.click(
                        generate_character_greeting_message,
                        inputs=[name, summary, personality, topic],
                        outputs=greeting_message
                    )

                # Alternate Greetings
                with gr.Row():
                    gr.Markdown("### Alternate Greetings")
                alternate_greetings = gr.Textbox(
                    label="Alternate Greetings",
                    lines=5,
                    placeholder="Enter each alternate greeting on a new line."
                )
                alternate_greetings_button = gr.Button("Generate Alternate Greetings with LLM")
                alternate_greetings_button.click(
                    generate_alternate_greetings,
                    inputs=[name, summary, personality, topic],
                    outputs=alternate_greetings
                )

                # Example messages
                with gr.Row():
                    with gr.Column():
                        switch_function_checkbox = gr.Checkbox(label="Use alternate example message generation", value=False)
                        example_messages = gr.Textbox(label="Example Messages", placeholder="Character example messages")
                    example_messages_button = gr.Button("Generate character example messages with LLM")
                    example_messages_button.click(
                        generate_example_messages,
                        inputs=[name, summary, personality, topic, switch_function_checkbox],
                        outputs=example_messages
                    )

                # Image row
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(interactive=True, label="Character Image", width=512, height=768)
                        process_image_button = gr.Button("Process Uploaded Image")
                        process_image_button.click(
                            process_uploaded_image,
                            inputs=[image_input],
                            outputs=[image_input]
                        )
                    with gr.Column():
                        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="(optional)")
                        avatar_prompt = gr.Textbox(label="Stable Diffusion Prompt", placeholder="(optional)")
                        combined_action_button = gr.Button("Update and Use Stable Diffusion Prompt")
                        combined_status = gr.Textbox(label="Status", interactive=False)
                        prompt_usage_output = gr.Textbox(label="Prompt Usage", interactive=False)
                        combined_action_button.click(
                            combined_avatar_prompt_action,
                            inputs=avatar_prompt,
                            outputs=[combined_status, prompt_usage_output]
                        )
                        generate_tags_button = gr.Button("Generate Tags and Set Prompt")
                        generate_tags_button.click(
                            generate_tags_and_set_prompt,
                            inputs=[image_input],
                            outputs=[avatar_prompt]
                        )
                        avatar_button = gr.Button("Generate Avatar (SDXL)")
                        avatar_button.click(
                            generate_character_avatar,
                            inputs=[name, summary, topic, negative_prompt, avatar_prompt, gender],
                            outputs=image_input
                        )

                # Export character
                with gr.Row():
                    export_json_button = gr.Button("Export as JSON")
                    exported_json = gr.Textbox(label="Exported JSON", lines=6)
                    export_json_button.click(
                        export_as_json,
                        inputs=[name, summary, personality, scenario, greeting_message, example_messages, alternate_greetings],
                        outputs=exported_json
                    )

                with gr.Row():
                    export_card_custom_button = gr.Button("Export as Custom PNG (All Fields)")
                    exported_card_image = gr.Image(label="Exported Character Card")
                    export_card_custom_button.click(
                        export_custom_png_with_json,
                        inputs=[name, summary, personality, scenario, greeting_message, example_messages, alternate_greetings],
                        outputs=exported_card_image
                    )

        # Import character
        with gr.Tab("Import character"):
            with gr.Column():
                with gr.Row():
                    import_card_input = gr.File(label="Upload Character Card (.png)", file_types=[".png"])
                    import_json_input = gr.File(label="Upload JSON File", file_types=[".json"])
                with gr.Row():
                    import_card_button = gr.Button("Import from Card")
                    import_json_button = gr.Button("Import from JSON")

                import_card_button.click(
                    lambda file: import_character_card(file.name) if file else None,
                    inputs=[import_card_input],
                    outputs=[name, summary, personality, scenario, greeting_message, example_messages]
                )
                import_json_button.click(
                    lambda file: import_character_json(file.name) if file else None,
                    inputs=[import_json_input],
                    outputs=[name, summary, personality, scenario, greeting_message, example_messages]
                )

        gr.HTML("""
        <div style='text-align: center; font-size: 20px;'>
            <p>
              <a style="text-decoration: none; color: inherit;" href="https://github.com/thijsi123/character-factory">
                Character Factory (SDXL Single-file, No Wiki, Infinite Alt Greetings)
              </a> 
              by 
              <a style="text-decoration: none; color: inherit;" href="https://github.com/Hukasx0">
                Hubert "Hukasx0" Kasperek
              </a>
              and forked by
              <a style="text-decoration: none; color: inherit;" href="https://github.com/thijsi123">
                Thijs
              </a>
            </p>
        </div>
        """)

    return webui

def main():
    load_sdxl_model()
    webui = create_webui()
    webui.launch(debug=True)

if __name__ == "__main__":
    main()
