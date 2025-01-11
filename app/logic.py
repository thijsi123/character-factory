# logic.py
###############################################################################
# 1. IMPORTS
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
import nltk

from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from diffusers import StableDiffusionXLPipeline
import onnxruntime as rt
import huggingface_hub

# If you have 'aichar' installed. Otherwise, remove references or adapt.
import aichar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# 2. GLOBALS
###############################################################################
global_url = "http://localhost:5001/api/v1/completions"
HF_TOKEN = "your_hf_token_here"

global_avatar_prompt = ""
processed_image_path = None

LOCAL_MODEL_PATH = r"G:\Documents\GitHub\character-factory\models\illustriousXLPersonalMerge_v30Noob10based.safetensors"
sd = None

# Path to your prompt modules
PROMPTS_DIR = r"G:\Documents\GitHub\character-factory\app\prompts"
if PROMPTS_DIR not in sys.path:
    sys.path.append(PROMPTS_DIR)

import prompt_name
import prompt_summary
import prompt_personality
import prompt_scenario
import prompt_greeting
import prompt_example
# (If you need prompt_nonnsfw_summary, import that too.)

def safe_get_dialogues(module, fallback):
    return getattr(module, "example_dialogue", fallback)

PROMPT_NAME_EXAMPLES = safe_get_dialogues(prompt_name, ["fallback name1"])
PROMPT_SUMMARY_EXAMPLES = safe_get_dialogues(prompt_summary, ["fallback summary1"])
PROMPT_PERSONALITY_EXAMPLES = safe_get_dialogues(prompt_personality, ["fallback personality1"])
PROMPT_SCENARIO_EXAMPLES = safe_get_dialogues(prompt_scenario, ["fallback scenario1"])
PROMPT_GREETING_EXAMPLES = safe_get_dialogues(prompt_greeting, ["fallback greeting1"])
PROMPT_EXAMPLE_DIALOGUES = safe_get_dialogues(prompt_example, ["fallback example1"])

###############################################################################
# 3. SETUP / UTILS
###############################################################################

def set_url(new_url):
    global global_url
    global_url = new_url.rstrip("/") + "/api/v1/completions"
    return f"URL set to {global_url}"

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
            "/s", "</s>", "<s>", "<|system|>", "<|assistant|>", "<|user|>", "<|char|>",
            "\n", "\nThijs:", "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            "\n", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
        "stopping_strings": [
            "/s", "</s>", "<s>", "<|system|>", "<|assistant|>", "<|user|>", "<|char|>",
            "\n", "\nThijs:", "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            "\n", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
    }
    try:
        resp = requests.post(global_url, json=request_body)
        resp.raise_for_status()
        result = resp.json().get("choices", [{}])[0].get("text", "")
        result = result.split("<")[0]
        return result
    except requests.RequestException as e:
        return f"Error: {e}"

def load_sdxl_model():
    global sd
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.error(f"Local model path does not exist: {LOCAL_MODEL_PATH}")
        sys.exit(1)
    print(f"Loading SDXL model from: {LOCAL_MODEL_PATH}")

    try:
        if torch.cuda.is_available():
            sd = StableDiffusionXLPipeline.from_single_file(LOCAL_MODEL_PATH, torch_dtype=torch.float16)
            sd.to("cuda")
            print("SDXL loaded in float16 on GPU.")
        else:
            sd = StableDiffusionXLPipeline.from_single_file(LOCAL_MODEL_PATH, torch_dtype=torch.float32)
            sd.to("cpu")
            print("SDXL loaded in float32 on CPU.")
    except Exception as e:
        logger.error(f"Failed loading model: {e}")
        sys.exit(1)

def input_none(text):
    return None if text == "" else text

###############################################################################
# 4. IMAGE CAPTIONING
###############################################################################
import onnxruntime as rt
import huggingface_hub

imagecaption_model = None
tags_df = None
target_size = None
current_model_repo = None

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

def download_model(model_repo, hf_token=None):
    logger.info(f"Attempting to download model from {model_repo}")
    try:
        csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME, use_auth_token=hf_token)
        model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME, use_auth_token=hf_token)
        return csv_path, model_path
    except Exception as e:
        logger.warning(f"Failed to download from {model_repo}: {e}")
        return None, None

def load_imagecaption_model(model_repo=None, hf_token=HF_TOKEN):
    global imagecaption_model, tags_df, target_size, current_model_repo
    if imagecaption_model is None or model_repo != current_model_repo:
        for repo in (MODEL_REPOS if model_repo is None else [model_repo]):
            csv_path, model_path = download_model(repo, hf_token)
            if csv_path and model_path:
                try:
                    tags_df = pd.read_csv(csv_path)
                    imagecaption_model = rt.InferenceSession(model_path)
                    _, target_size_, _, _ = imagecaption_model.get_inputs()[0].shape
                    target_size = target_size_
                    current_model_repo = repo
                    return imagecaption_model, tags_df, target_size
                except Exception as e:
                    logger.warning(f"Error loading from {repo}: {e}")

    raise ValueError("No model loaded for image captioning")

def prepare_image(image, target_size_):
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        pil = image.convert("RGB")

    pil = pil.resize((target_size_, target_size_), Image.BICUBIC)
    arr = np.array(pil, dtype=np.float32)[:, :, ::-1]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_tags(img_array):
    global imagecaption_model
    input_name = imagecaption_model.get_inputs()[0].name
    preds = imagecaption_model.run(None, {input_name: img_array})[0]
    return preds

def get_sorted_general_strings(image, model_repo=None, hf_token=HF_TOKEN, threshold=0.35):
    global imagecaption_model, tags_df, target_size
    if imagecaption_model is None:
        imagecaption_model, tags_df, target_size = load_imagecaption_model(model_repo, hf_token)

    if isinstance(image, str):
        pil_image = Image.open(image)
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        pil_image = image

    arr = prepare_image(pil_image, target_size)
    preds = predict_tags(arr)

    names = tags_df['name'].tolist()
    all_tags = list(zip(names, preds[0].astype(float)))
    filtered = [(t, v) for (t, v) in all_tags if v > threshold]
    filtered = dict(filtered)
    sorted_tags = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    final_tags = [x[0] for x in sorted_tags]
    return ", ".join(final_tags)

###############################################################################
# 5. IMPORT/EXPORT
###############################################################################
def import_character_json(path):
    if path is not None:
        ch = aichar.load_character_json_file(path)
        if ch.name:
            return (
                ch.name, ch.summary, ch.personality,
                ch.scenario, ch.greeting_message, ch.example_messages
            )
        else:
            raise ValueError("Imported JSON has no character name.")

def import_character_card(path):
    if path is not None:
        ch = aichar.load_character_card_file(path)
        if ch.name:
            return (
                ch.name, ch.summary, ch.personality,
                ch.scenario, ch.greeting_message, ch.example_messages
            )
        else:
            raise ValueError("Imported card has no character name.")

def export_as_json(name, summary, personality, scenario, greeting_message, example_messages):
    ch = aichar.create_character(
        name, summary, personality, scenario, greeting_message,
        example_messages, image_path=""
    )
    return ch.export_neutral_json()

def export_character_card(name, summary, personality, scenario, greeting_message, example_messages):
    global processed_image_path
    cname = name.replace(" ", "_")
    base_dir = os.path.join("characters", cname)
    os.makedirs(base_dir, exist_ok=True)

    if processed_image_path and os.path.exists(processed_image_path):
        img_path = processed_image_path
    else:
        img_path = "characters/uploaded_character/uploaded_character.png"
    if not os.path.exists(img_path):
        logger.error("Image not found. Creating default.")
        im = Image.new("RGB", (256,256), color=(73,109,137))
        img_path = os.path.join(base_dir, f"{cname}.png")
        im.save(img_path)

    try:
        ch = aichar.create_character(
            name, summary, personality, scenario,
            greeting_message, example_messages,
            image_path=img_path
        )
        out_path = os.path.join(base_dir, f"{cname}.card.png")
        ch.export_neutral_card_file(out_path)
        if os.path.exists(out_path):
            return out_path
        else:
            return None
    except Exception as e:
        logger.exception(f"Error exporting card: {e}")
        return None

###############################################################################
# 6. CHARACTER GENERATION (Using Prompt Modules)
###############################################################################

def clean_output_brackets(raw_output, character_name):
    def ensure_double_brackets(m):
        return "{{" + m.group(1) + "}}"
    c = re.sub(r"\{{3,}(char|user)\}{3,}", ensure_double_brackets, raw_output)
    c = re.sub(r"\{{1,2}(char|user)\}{1,2}", ensure_double_brackets, c)
    if character_name:
        c = re.sub(r"\b" + re.escape(character_name) + r"\b", "{{char}}", c)
    c = re.sub(r'\*\s*"', '"', c)
    c = re.sub(r'"\s*\*', '"', c)
    return c

def generate_character_name(topic, gender, name, surname_checkbox):
    ex = random.choice(PROMPT_NAME_EXAMPLES)
    gender_txt = f"Character gender: {gender}." if gender else ""
    surname_txt = "Add Surname" if surname_checkbox else ""
    prompt = (
        ex
        + f"\n<|user|> Generate a random character first name. "
        f"Topic: {topic}. {gender_txt} {surname_txt}\n</s>\n<|assistant|>"
    )
    out = send_message(prompt)
    out = re.sub(r"[^a-zA-Z0-9_ -]", "", out).strip()
    return out

def generate_character_summary(name, topic, gender):
    ex = random.choice(PROMPT_SUMMARY_EXAMPLES)
    app = global_avatar_prompt
    user_prompt = (
        f"{ex}\n<|user|> Create a longer description for a character named: {name}, "
        f"Topic: {topic}, "
        f"{'Character gender: ' + gender + '.' if gender else ''} "
    )
    if app:
        user_prompt += f"This character has an appearance of {app}. Use those tags. "
    user_prompt += "Make the summary descriptive. </s>\n<|assistant|>"
    out = send_message(user_prompt)
    return out.strip()

def generate_character_personality(name, summary, topic):
    ex = random.choice(PROMPT_PERSONALITY_EXAMPLES)
    user_prompt = (
        f"{ex}\n<|user|> Describe the personality of {name}. "
        f"Their characteristics: {summary}. Tailor them to {topic}.\n</s>\n<|assistant|>"
    )
    return send_message(user_prompt).strip()

def generate_character_scenario(summary, personality, topic):
    ex = random.choice(PROMPT_SCENARIO_EXAMPLES)
    user_prompt = (
        f"{ex}\n<|user|> Write a scenario for a roleplay with {{char}} & {{user}}. "
        f"{{char}} has these traits: {summary}, {personality}. "
        f"Theme: {topic}. No dialogues.\n</s>\n<|assistant|>"
    )
    raw = send_message(ex + user_prompt)
    return raw.strip()

def generate_character_greeting_message(name, summary, personality, topic):
    ex = random.choice(PROMPT_GREETING_EXAMPLES)
    user_prompt = (
        ex
        + "\n<|user|> Create the first message from "
        + f"{name} who has {summary},{personality}, "
        + f"theme {topic}, greeting {{user}}.\n</s>\n<|assistant|>"
    )
    raw = send_message(user_prompt).strip()
    return clean_output_brackets(raw, name)

def generate_example_messages(name, summary, personality, topic, switch_checkbox):
    ex = random.choice(PROMPT_EXAMPLE_DIALOGUES)
    user_prompt = (
        f"{ex}\n<|user|> Create a dialogue between {{user}} and {{char}}. "
        f"{{char}} is {name} with traits {summary},{personality}, "
        f"theme: {topic}. Make it interesting.\n</s>\n<|assistant|>"
    )
    raw = send_message(ex + user_prompt).strip()
    return clean_output_brackets(raw, name)

###############################################################################
# 7. IMAGE GENERATION
###############################################################################

def fetch_avatar_prompt(summary, topic, gender):
    # We can pick from summary or from name modules as needed
    ex = random.choice(PROMPT_SUMMARY_EXAMPLES)
    user_prompt = f"""
<|system|>
You are a text generation tool returning only tags describing the character's appearance.
</s>
<|user|>:
create a prompt that lists the appearance of a character 
whose gender is {gender}, 
whose summary is: {summary}.
Theme: {topic}
</s>
<|assistant|>:
"""
    out = send_message(user_prompt)
    return out.strip()

def generate_character_avatar(name, summary, topic, negative_prompt, avatar_prompt, gender):
    if not avatar_prompt.strip():
        avatar_prompt = fetch_avatar_prompt(summary, topic, gender)
    style_prefix = "anime, 2d, " if ("anime" in summary.lower() or "anime" in topic.lower()) else "realistic, 3d, "
    final_prompt = style_prefix + avatar_prompt.strip()
    return image_generate(name, final_prompt, input_none(negative_prompt))

def image_generate(character_name, prompt, neg):
    global processed_image_path
    default_neg = (
        "worst quality, low quality, blurry, text, watermark, logo, banner, "
        "duplicate, poorly drawn, bad anatomy, missing limbs, extra limbs, "
        "ugly face, multiple faces, etc."
    )
    negp = default_neg + " " + (neg or "")
    try:
        result = sd(prompt=prompt, negative_prompt=negp, height=1152, width=768, num_inference_steps=50, guidance_scale=5)
        im = result.images[0]
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

    safe_name = character_name.replace(" ", "_")
    out_dir = os.path.join("characters", safe_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name}.png")
    im.save(out_path, "PNG")

    arr = np.array(im)
    process_uploaded_image(arr, save_as=f"{safe_name}_uploaded.png", out_dir=out_dir)
    return im

def process_uploaded_image(arr, save_as="uploaded.png", out_dir="characters/uploaded_character"):
    global processed_image_path
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(arr, np.ndarray):
        pil_im = Image.fromarray(np.uint8(arr)).convert("RGB")
    else:
        pil_im = arr

    full_path = os.path.join(out_dir, save_as)
    pil_im.save(full_path)
    processed_image_path = full_path
    logger.info(f"Uploaded image saved at: {processed_image_path}")
    return pil_im

###############################################################################
# 8. INFINITE ALTERNATE GREETINGS
###############################################################################

def init_greeting_lines():
    return ["Greeting line 1 (edit me)"]

def insert_greeting_line(g_lines, idx):
    new_line = "New alt greeting line"
    g_lines.insert(idx+1, new_line)
    return g_lines

def remove_greeting_line(g_lines, idx):
    if 0 <= idx < len(g_lines):
        g_lines.pop(idx)
    return g_lines

def update_greeting_line(g_lines, idx, new_text):
    if 0 <= idx < len(g_lines):
        g_lines[idx] = new_text
    return g_lines

def greeting_lines_to_text(g_lines):
    return "\n".join(g_lines)
