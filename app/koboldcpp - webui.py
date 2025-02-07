###############################################################################
# 1. ENVIRONMENT SETUP BEFORE IMPORTS
###############################################################################
import os

# Set CUDA_VISIBLE_DEVICES before importing any CUDA-related libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Replace "1" with "0" or the desired GPU index

###############################################################################
# 2. IMPORTS & GLOBAL VARIABLES
###############################################################################
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

from PIL import Image
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# If you have 'aichar' installed. Otherwise, remove or replace references.
import aichar

# For SDXL
from diffusers import StableDiffusionXLPipeline
import onnxruntime as rt
import huggingface_hub

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# 3. VERIFY CUDA SETUP
###############################################################################

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    logger.info(f"Using CUDA device {current_device}: {device_name}")
else:
    logger.info("CUDA is not available. Using CPU.")

###############################################################################
# 4. SETUP: PATHS, IMPORT PROMPTS, GLOBALS
###############################################################################

# Your local prompt directory, e.g. G:\Documents\GitHub\character-factory\app\prompts
PROMPTS_DIR = r"G:\Documents\GitHub\character-factory\app\prompts"
# Convert to an absolute path; helps Python find the modules
PROMPTS_DIR = os.path.abspath(PROMPTS_DIR)

# Add PROMPTS_DIR to Python's module search path
if PROMPTS_DIR not in sys.path:
    sys.path.append(PROMPTS_DIR)

# Now import the prompt modules
import prompt_name
import prompt_summary
import prompt_personality
import prompt_scenario
import prompt_greeting
import prompt_example

# If you have a separate prompt_nonnsfw_summary, import that if needed
# import prompt_nonnsfw_summary

# We'll read their `example_dialogue` lists
PROMPT_NAME_EXAMPLES = getattr(prompt_name, "example_dialogue", ["Fallback name1", "Fallback name2"])
PROMPT_SUMMARY_EXAMPLES = getattr(prompt_summary, "example_dialogue", ["Fallback summary1", "Fallback summary2"])
PROMPT_PERSONALITY_EXAMPLES = getattr(prompt_personality, "example_dialogue",
                                      ["Fallback personality1", "Fallback personality2"])
PROMPT_SCENARIO_EXAMPLES = getattr(prompt_scenario, "example_dialogue", ["Fallback scenario1", "Fallback scenario2"])
PROMPT_GREETING_EXAMPLES = getattr(prompt_greeting, "example_dialogue", ["Fallback greeting1", "Fallback greeting2"])
PROMPT_EXAMPLE_DIALOGUES = getattr(prompt_example, "example_dialogue", ["Fallback example1", "Fallback example2"])

# Default global URL for LLM completions
global_url = "http://localhost:5001/api/v1/completions"

# Hugging Face token for imagecaption (change or remove if needed)
HF_TOKEN = "your_hf_token_here"

# Global variables for the UI flow
global_avatar_prompt = ""
processed_image_path = None

# Path to your local SDXL single-file .safetensors model
# Example:
LOCAL_MODEL_PATH = r"G:\Documents\GitHub\character-factory\models\illustriousXLPersonalMerge_v30Noob10based.safetensors"

# Our loaded SDXL pipeline
sd = None


###############################################################################
# 5. UTILS / SHARED
###############################################################################

def process_url(url: str):
    """
    Sets the global_url for LLM completions.
    """
    global global_url
    global_url = url.rstrip("/") + "/api/v1/completions"
    return f"URL Set: {global_url}"


def send_message(prompt: str, max_new_tokens: int = 2048, max_tokens: int = 2048) -> str:
    """
    Sends the given prompt to your local or remote LLM API specified by global_url.

    Parameters:
    - prompt (str): The prompt to send to the LLM.
    - max_new_tokens (int): The maximum number of new tokens to generate. Default is 2048.
    - max_tokens (int): The maximum total number of tokens in the response. Default is 2048.

    Returns:
    - str: The generated response from the LLM or an error message.
    """
    global global_url
    if not global_url:
        return "Error: URL not set."

    request_body = {
        "prompt": prompt,
        "max_length": 8192,
        "max_new_tokens": max_new_tokens,  # Updated parameter
        "max_tokens": max_tokens,  # Updated parameter
        "max_content_length": 8192,
        "do_sample": True,
        "temperature": 1,
        "typical_p": 1,
        "repetition_penalty": 1.1,
        "guidance_scale": 1,
        "sampler_seed": -1,
        "stop": [
            "/s", "</s>", "<s>", "<|system|>", "<|assistant|>", "<|user|>", "<|char|>",
            r"\n", "\nThijs:", "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
        "stopping_strings": [
            "/s", "</s>", "<s>", "<|system|>", "<|assistant|>", "<|user|>", "<|char|>",
            r"\n", "\nThijs:", "<|END_OF_TURN_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
            r"\n", "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>New Roleplay:<|END_OF_TURN_TOKEN|>"
        ],
    }

    try:
        response = requests.post(global_url, json=request_body)
        response.raise_for_status()
        result = response.json().get("choices", [{}])[0].get("text", "")
        # Remove any trailing content after "<"
        result = result.split("<")[0]
        return result.strip()
    except requests.RequestException as e:
        return f"Error sending request: {e}"


def input_none(text):
    return None if (text == "") else text


def combined_avatar_prompt_action(prompt: str):
    global global_avatar_prompt
    global_avatar_prompt = prompt
    return "Avatar prompt updated!", f"Using avatar prompt: {global_avatar_prompt}"


###############################################################################
# 6. LOADING SDXL FROM A SINGLE-FILE
###############################################################################

from diffusers import StableDiffusionXLPipeline


def load_sdxl_model():
    """
    Loads a single-file .safetensors SDXL model using StableDiffusionXLPipeline.
    If your model is not truly SDXL or lacks the necessary config, this may fail.
    """
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
            ).to("cuda:0")  # Explicitly use the first visible GPU
            print("SDXL model loaded in float16 on GPU 0.")
        else:
            sd = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32
            ).to("cpu")
            print("SDXL model loaded in float32 on CPU.")
    except Exception as e:
        logger.error(f"Failed to load local model from {model_path}: {e}")
        sys.exit(1)


###############################################################################
# 7. IMAGE CAPTIONING (ONNX)
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
                    input_shape = imagecaption_model.get_inputs()[0].shape
                    if len(input_shape) == 4:
                        _, _, target_size_, _ = input_shape
                    elif len(input_shape) == 3:
                        _, target_size_, _ = input_shape
                    else:
                        target_size_ = 224  # Default fallback
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
# 8. WIKI
###############################################################################
nltk.download('punkt', quiet=True)


def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('div', class_='mw-parser-output')
        if main_content:
            text_content = []
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text_content.append(element.get_text())
            content = ' '.join(text_content)
        else:
            content = soup.get_text()
        if not content.strip():
            raise ValueError("No text content found on the page")
        return content
    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error: {str(e)}"


class AdvancedVectorStorage:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []
        self.texts = []

    def add_text(self, text):
        chunks = self.chunk_text(text)
        for c in chunks:
            self.texts.append(c)
            self.vectors.append(self.model.encode(c))

    def chunk_text(self, text, max_length=200):
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        for w in words:
            if current_len + len(w) > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(w)
            current_len += len(w) + 1
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def query(self, query_text, top_k=5):
        query_vector = self.model.encode(query_text)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.texts[i], similarities[i]) for i in top_indices]


def generate_character_summary_from_fandom(fandom_url, character_name=None, topic=None, gender=None, appearance=None):
    content = scrape_website(fandom_url)
    if content.startswith("Error") or content.startswith("Unexpected"):
        return content

    storage = AdvancedVectorStorage()
    storage.add_text(content)

    if not character_name:
        character_name = fandom_url.split('/')[-1].replace('_', ' ')
    if not topic:
        topic = fandom_url.split('/')[2].split('.')[0].capitalize()

    queries = [
        f"What is {character_name}'s appearance and distinctive features?",
        f"What are {character_name}'s personality traits and characteristics?",
        f"What is {character_name}'s role or significance in {topic}?"
    ]
    relevant_info = []
    for q in queries:
        results = storage.query(q, top_k=3)
        relevant_info.extend([t for t, _ in results])

    context = " ".join(relevant_info)
    if not context:
        return f"Error: No relevant info about {character_name} from the URL."

    base_prompt = (
            f"Create a summary for {character_name} in the context of topic '{topic}'. "
            f"Gender: {gender if gender else 'Unknown'}. "
            f"Appearance: {appearance if appearance else 'None provided'}. "
            "Also include relevant info from the wiki:\n" + context
    )

    final_summary = send_message(base_prompt).strip()
    return final_summary


###############################################################################
# 9. IMPORT/EXPORT
###############################################################################

def find_image_path():
    possible_paths = ["./app2/image.png", "./image.png"]
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
    if json_path is not None:
        character = aichar.load_character_json_file(json_path.name)
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
    if card_path is not None:
        character = aichar.load_character_card_file(card_path.name)
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


def export_as_json(name, summary, personality, scenario, greeting_message, example_messages):
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=""
    )
    return character.export_neutral_json()


def export_character_card(name, summary, personality, scenario, greeting_message, example_messages):
    global processed_image_path
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
        img = Image.new("RGB", (256, 256), color=(73, 109, 137))
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
            return Image.open(card_path)
        else:
            logger.error(f"Character card file not created: {card_path}")
            return None
    except Exception as e:
        logger.exception(f"Error creating character card: {str(e)}")
        return None


###############################################################################
# 10. CHARACTER GENERATION
#    *Now uses the lists from the prompt_*.py files instead of hardcoded
###############################################################################

def generate_character_name(topic, gender, name, surname_checkbox):
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
    from_prompt = random.choice(PROMPT_SUMMARY_EXAMPLES)
    user_prompt = (
        f"{from_prompt}\n"
        f"<|user|> Create a longer description for {character_name}, "
        f"Topic: {topic}, "
        f"{'Character gender: ' + gender + '.' if gender else ''} "
    )
    # If we have an avatar prompt in global_avatar_prompt, add it
    global global_avatar_prompt
    if global_avatar_prompt:
        user_prompt += f"This character has an appearance of {global_avatar_prompt}. Use those tags. "

    user_prompt += "Make the summary descriptive. </s>\n<|assistant|>"

    output = send_message(user_prompt).strip()
    return output


def generate_character_personality(character_name, character_summary, topic):
    from_prompt = random.choice(PROMPT_PERSONALITY_EXAMPLES)
    user_prompt = (
        f"{from_prompt}\n"
        f"<|user|> Describe the personality of {character_name}. "
        f"Characteristics: {character_summary}. Tailor to {topic}.\n</s>\n<|assistant|>"
    )
    return send_message(user_prompt).strip()


def generate_character_scenario(character_summary, character_personality, topic):
    from_prompt = random.choice(PROMPT_SCENARIO_EXAMPLES)
    user_prompt = (
        f"{from_prompt}\n"
        f"<|user|> Write a scenario for a roleplay with {{char}} & {{user}}. "
        f"{{char}} has these traits: {character_summary}, {character_personality}. "
        f"Theme: {topic}. No dialogues.\n</s>\n<|assistant|>"
    )
    return send_message(user_prompt).strip()


def generate_character_greeting_message(character_name, character_summary, character_personality, topic):
    from_prompt = random.choice(PROMPT_GREETING_EXAMPLES)
    user_prompt = (
        f"{from_prompt}\n"
        f"<|user|> Create the first message from {character_name}, who has {character_summary},"
        f"{character_personality}, theme {topic}, greeting {{user}}.\n</s>\n<|assistant|>"
    )
    raw_output = send_message(user_prompt).strip()
    return clean_output_brackets(raw_output, character_name)


def generate_example_messages(character_name, character_summary, character_personality, topic,
                              switch_function_checkbox):
    from_prompt = random.choice(PROMPT_EXAMPLE_DIALOGUES)
    user_prompt = (
        f"{from_prompt}\n"
        f"<|user|> Create a dialogue between {{user}} and {{char}}. "
        f"{{char}} is {character_name} with traits {character_summary}, {character_personality}, "
        f"theme: {topic}. Make it interesting.\n</s>\n<|assistant|>"
    )
    raw_output = send_message(user_prompt).strip()
    return clean_output_brackets(raw_output, character_name)


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


###############################################################################
# 11. IMAGE GENERATION
###############################################################################

def fetch_avatar_prompt(character_name, character_summary, topic, gender):
    # Escape any curly braces in character_summary to prevent .format() from misinterpreting them
    escaped_summary = character_summary.replace('{', '{{').replace('}', '}}')

    example_dialogue = """
<|system|>
You are a text generation tool, returning only tags describing the character's appearance with just danbooru like tags separated by commas, no sentences, no clip, pure danbooru tags. Do only describe the appearance of the character, do not add tags for hobbies, quirks, personality, etc.
</s>
<|user|>: Describe the appearance of Jamie Hale. Their characteristics Jamie Hale is a tall man standing at 6 feet with a confident and commanding presence. He has short, dark hair, piercing blue eyes, and an impeccable sense of style, often seen in tailored suits. Jamie exudes charisma and carries himself with an air of authority that draws people to him.</s>
<|assistant|>: 1boy, tall, short dark hair, piercing blue eyes, tailored suits, </s>
<|user|>: Describe the appearance of Mr. Fluffy. Their characteristics Mr. Fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework.</s>
<|assistant|>: 1cat, fat, fluffy, black and white fur, lying on lap</s>
<|user|>: Describe the appearance of {character_name}. Their characteristics {gender} {escaped_summary}</s>
<|assistant|>: 
""".format(
        character_name=character_name,
        gender=gender,
        escaped_summary=escaped_summary,
        topic=topic
    )

    # Set max_new_tokens and max_tokens to 100 for avatar prompt generation
    raw_prompt = send_message(example_dialogue, max_new_tokens=60, max_tokens=60)
    return raw_prompt.strip()


def generate_character_avatar(character_name, character_summary, topic, negative_prompt, avatar_prompt, gender,
                              style_selection):
    """
    Generates a character avatar using the loaded SDXL pipeline.
    """
    if not avatar_prompt.strip():
        avatar_prompt = fetch_avatar_prompt(character_name, character_summary, topic, gender)

    # Determine the style prefix based on user selection (Radio Buttons)
    if style_selection.lower() == "anime":
        style_prefix = "anime, "
    else:
        style_prefix = "realistic, "

    final_prompt = style_prefix + avatar_prompt.strip()

    return image_generate(character_name, final_prompt, input_none(negative_prompt))


def image_generate(character_name, prompt, negative_prompt):
    global processed_image_path

    default_negative = (
        "worst quality, low quality, blurry, text, watermark, logo, banner, "
        "duplicate, poorly drawn, bad anatomy, missing limbs, extra limbs, "
        "ugly face, multiple faces,"
    )
    negative_prompt = default_negative + " " + (negative_prompt or "")

    try:
        # Make sure 'sd' is loaded from load_sdxl_model() using StableDiffusionXLPipeline
        result = sd(
            prompt=prompt,  # This prompt includes the style_prefix
            negative_prompt=negative_prompt,
            height=1152,
            width=768,
            num_inference_steps=30,
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
# 12. UI / MAIN
###############################################################################

def generate_tags_and_set_prompt(image):
    tags = get_sorted_general_strings(image)
    return tags


def create_webui():
    with gr.Blocks() as webui:
        gr.Markdown("# Character Factory WebUI (SDXL Single-file)")
        gr.Markdown("## KOBOLD MODE")

        with gr.Row():
            url_input = gr.Textbox(label="Enter URL", value="http://127.0.0.1:5001")
            submit_button = gr.Button("Set URL")
        output = gr.Textbox(label="URL Status")
        submit_button.click(process_url, inputs=url_input, outputs=output)

        with gr.Tab("Edit character"):
            topic = gr.Textbox(label="Topic", placeholder="e.g. Fantasy, Anime, etc.")
            gender = gr.Textbox(label="Gender", placeholder="M, F, etc.")

            with gr.Column():
                # Name
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

                # UI placeholders for combined statuses
                combined_status = gr.Textbox(label="Status", interactive=False)
                prompt_usage_output = gr.Textbox(label="Prompt Usage", interactive=False)
                combined_action_button = gr.Button("Update and Use Stable Diffusion Prompt")

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

                # Greeting message + plus/minus buttons
                with gr.Row():
                    greeting_message = gr.Textbox(label="Greeting Message", placeholder="Character greeting message")
                    greeting_message_button = gr.Button("Generate character greeting message with LLM")
                    greeting_message_button.click(
                        generate_character_greeting_message,
                        inputs=[name, summary, personality, topic],
                        outputs=greeting_message
                    )

                # Example messages
                with gr.Row():
                    switch_function_checkbox = gr.Checkbox(label="Use alternate example message generation",
                                                           value=False)
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
                        # Added Style Selection Radio Buttons
                        style_selection = gr.Radio(
                            choices=["Anime", "Realistic"],
                            value="Anime",  # Default value
                            label="Select Style",
                            interactive=True
                        )

                        negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="(optional)")
                        avatar_prompt = gr.Textbox(label="Stable Diffusion Prompt", placeholder="(optional)")
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
                            inputs=[name, summary, topic, negative_prompt, avatar_prompt, gender, style_selection],
                            outputs=image_input
                        )

        # Additional tabs: Wiki Character, Import, Export, etc.
        with gr.Tab("Wiki Character"):
            with gr.Column():
                wiki_url = gr.Textbox(label="Fandom Wiki URL", placeholder="Paste character wiki URL")
                wiki_character_name = gr.Textbox(label="Character Name", placeholder="(Optional override)")
                wiki_topic = gr.Textbox(label="Topic/Series", placeholder="(Optional override, e.g. 'Zelda')")
                wiki_gender = gr.Textbox(label="Gender (optional)")
                wiki_appearance = gr.Textbox(label="Appearance (optional)")

                wiki_generate_button = gr.Button("Generate Character Summary from Wiki")
                wiki_summary_output = gr.Textbox(label="Generated Character Summary", lines=10)

                wiki_generate_button.click(
                    generate_character_summary_from_fandom,
                    inputs=[wiki_url, wiki_character_name, wiki_topic, wiki_gender, wiki_appearance],
                    outputs=wiki_summary_output
                )

                wiki_update_button = gr.Button("Update Character with Wiki Summary")
                wiki_update_button.click(
                    lambda wname, wsummary: (wname, wsummary),
                    inputs=[wiki_character_name, wiki_summary_output],
                    outputs=[name, summary]
                )

        with gr.Tab("Import character"):
            with gr.Column():
                with gr.Row():
                    import_card_input = gr.File(label="Upload Character Card (.png)", file_types=[".png"])
                    import_json_input = gr.File(label="Upload JSON File", file_types=[".json"])
                with gr.Row():
                    import_card_button = gr.Button("Import from Card")
                    import_json_button = gr.Button("Import from JSON")

                import_card_button.click(
                    import_character_card,
                    inputs=[import_card_input],
                    outputs=[name, summary, personality, scenario, greeting_message, example_messages]
                )
                import_json_button.click(
                    import_character_json,
                    inputs=[import_json_input],
                    outputs=[name, summary, personality, scenario, greeting_message, example_messages]
                )

        with gr.Tab("Export character"):
            with gr.Column():
                with gr.Row():
                    export_image = gr.Image(width=512, height=512)
                    export_json_textbox = gr.JSON()

                with gr.Row():
                    export_card_button = gr.Button("Export as Character Card")
                    export_json_button = gr.Button("Export as JSON")

                    export_card_button.click(
                        export_character_card,
                        inputs=[name, summary, personality, scenario, greeting_message, example_messages],
                        outputs=export_image
                    )
                    export_json_button.click(
                        export_as_json,
                        inputs=[name, summary, personality, scenario, greeting_message, example_messages],
                        outputs=export_json_textbox
                    )

        gr.HTML("""
        <div style='text-align: center; font-size: 20px;'>
            <p>
              <a style="text-decoration: none; color: inherit;" href="https://github.com/thijsi123/character-factory">
                Character Factory (SDXL Single-file)
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
    load_sdxl_model()  # Load the single-file .safetensors (SDXL)
    webui = create_webui()
    webui.launch(debug=True)


if __name__ == "__main__":
    main()
