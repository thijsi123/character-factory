# import_export.py

import aicharl as aichar
from PIL import Image
import os
import logging
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable to store the path of any processed image (set by your UI code)
processed_image_path = None

def find_image_path():
    """
    Optional helper function to find a default image for the character.
    """
    possible_paths = [
        "./app2/image.png",
        "./image.png"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

image_path = find_image_path()
if image_path:
    print(f"Image found at: {image_path}")
else:
    print("Image not found in any of the specified locations.")

# ----------------- IMPORT LOGIC ----------------- #

def import_character_json(json_path):
    """
    Imports a character from a JSON file using aicharl.load_character_json_file().
    """
    if json_path is not None:
        character = aichar.load_character_json_file(json_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.first_greeting_message,  # only if you want both
                character.greeting_message,        # ...
                character.example_messages,
                *character.alternate_greetings,
            )
        raise ValueError("Error importing from JSON: check the file for correctness.")

def import_character_card(card_path):
    """
    Imports a character from a PNG character card using aicharl.load_character_card_file().
    """
    if card_path is not None:
        character = aichar.load_character_card_file(card_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.first_greeting_message,
                character.greeting_message,
                character.example_messages,
                *character.alternate_greetings,
            )
        raise ValueError("Error importing from card file: check the file for correctness.")


# ----------------- EXPORT LOGIC: JSON ----------------- #

def export_as_json(
    name,               # 1
    summary,            # 2
    personality,        # 3
    scenario,           # 4
    first_greeting_message,  # 5
    greeting_message,        # 6
    example_messages,        # 7
    *alternate_greetings     # 8+
):
    """
    Creates a character from UI fields and returns the JSON string in 'neutral' format.
    Ensure the param order matches your Gradio UI .click(inputs=[...]).
    """
    alt_greetings_list = [g for g in alternate_greetings if g.strip()]
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        first_greeting_message=first_greeting_message,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=None,  # or use 'image_path' if you want a default
        alternate_greetings=alt_greetings_list,
    )
    return character.export_neutral_json()


# ----------------- EXPORT LOGIC: CARD ----------------- #

def export_character_card(
    name,                   # 1
    summary,                # 2
    personality,            # 3
    scenario,               # 4
    first_greeting_message, # 5
    greeting_message,       # 6
    example_messages,       # 7
    *alternate_greetings
):
    """
    Creates a character from the UI fields, then exports as a .card.png file
    with base64-encoded metadata inside. Also uses the global processed_image_path
    if it exists, else a default image.
    """
    global processed_image_path  # So we can check/modify that global variable
    logger.debug(f"Exporting character card for: {name}")

    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"
    os.makedirs(base_path, exist_ok=True)

    # If a processed image was saved, use it
    if processed_image_path and os.path.exists(processed_image_path):
        local_image_path = processed_image_path
        logger.debug(f"Using processed image: {local_image_path}")
    else:
        # Otherwise, use or create a default image
        local_image_path = "characters/uploaded_character/uploaded_character.png"
        if not os.path.exists(local_image_path):
            logger.debug(f"No default image found; creating new 256x256 placeholder.")
            os.makedirs("characters/uploaded_character", exist_ok=True)
            img = Image.new('RGB', (256, 256), color=(73, 109, 137))
            img.save(local_image_path)

    alt_greetings_list = [g for g in alternate_greetings if g.strip()]
    logger.debug(f"Number of alt greetings: {len(alt_greetings_list)}")

    # Create the character
    try:
        character = aichar.create_character(
            name=name,
            summary=summary,
            personality=personality,
            scenario=scenario,
            first_greeting_message=first_greeting_message,
            greeting_message=greeting_message,
            example_messages=example_messages,
            image_path=local_image_path,
            alternate_greetings=alt_greetings_list,
        )

        # Export the .card.png
        card_path = f"{base_path}{character_name}.card.png"
        logger.debug(f"Exporting character card to: {card_path}")
        character.export_neutral_card_file(card_path)

        if os.path.exists(card_path):
            # Return the path or an Image object
            return Image.open(card_path)
        else:
            logger.error(f"Character card file not created: {card_path}")
            return None

    except Exception as e:
        logger.exception(f"Error creating character card: {str(e)}")
        return None
