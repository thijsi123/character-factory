import aichar
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_image_path():
    possible_paths = [
        "./app2/image.png",
        "./image.png"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None  # Return None if no image is found

image_path = find_image_path()

if image_path:
    print(f"Image found at: {image_path}")
else:
    print("Image not found in any of the specified locations.")


def import_character_json(json_path):
    print(json_path)
    if json_path is not None:
        character = aichar.load_character_json_file(json_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
        raise ValueError(
            "Error when importing character data from a JSON file. Validate the file. Check the file for correctness and try again")  # nopep8


def import_character_card(card_path):
    print(card_path)
    if card_path is not None:
        character = aichar.load_character_card_file(card_path)
        if character.name:
            gr.Info("Character data loaded successfully")
            return (
                character.name,
                character.summary,
                character.personality,
                character.scenario,
                character.greeting_message,
                character.example_messages,
            )
        raise ValueError(
            "Error when importing character data from a character card file. Check the file for correctness and try again")  # nopep8


def export_as_json(
        name, summary, personality, scenario, greeting_message, example_messages
):
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path="",
    )
    return character.export_neutral_json()


# Global variable to store the path of the processed image
processed_image_path = None


def export_character_card(name, summary, personality, scenario, greeting_message, example_messages):
    global processed_image_path  # Access the global variable

    # Prepare the character's name and base path
    character_name = name.replace(" ", "_")
    base_path = f"characters/{character_name}/"

    logger.debug(f"Base path: {base_path}")

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    if processed_image_path is not None and os.path.exists(processed_image_path):
        # If an image has been processed and exists, use it
        image_path = processed_image_path
        logger.debug(f"Using processed image: {image_path}")
    else:
        # If no image has been processed or the file doesn't exist, use a default image
        image_path = f"characters/uploaded_character/uploaded_character.png"
        logger.debug(f"Using default image: {image_path}")

    # Check if the image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        # Create a simple default image
        img = Image.new('RGB', (256, 256), color=(73, 109, 137))
        image_path = f"{base_path}{character_name}.png"
        img.save(image_path)
        logger.debug(f"Created new default image: {image_path}")

    # Create the character with the appropriate image
    try:
        character = aichar.create_character(
            name=name,
            summary=summary,
            personality=personality,
            scenario=scenario,
            greeting_message=greeting_message,
            example_messages=example_messages,
            image_path=image_path
        )

        # Export the character card
        card_path = f"{base_path}{character_name}.card.png"
        logger.debug(f"Exporting character card to: {card_path}")
        character.export_neutral_card_file(card_path)

        if os.path.exists(card_path):
            return Image.open(card_path)
        else:
            logger.error(f"Character card file not created: {card_path}")
            return None
    except Exception as e:
        logger.exception(f"Error creating character card: {str(e)}")
        return None
