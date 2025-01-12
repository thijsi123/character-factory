# aicharl.py
"""
A Python library to create/load AI character definitions in various formats
(JSON, YAML, PNG 'character cards'), for use with SillyTavern, TavernAI, etc.
"""

import json
import yaml
import base64
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO
from PIL import Image, PngImagePlugin

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


############################################################
# CharacterClass definition
############################################################

class CharacterClass:
    """
    Represents an AI character with fields like name, summary, personality, etc.,
    and methods to export/import from JSON, YAML, or PNG card files.
    """

    def __init__(
        self,
        name: str,
        summary: str,
        personality: str,
        scenario: str,
        first_greeting_message: str,
        greeting_message: str,
        example_messages: str,
        image_path: Optional[str] = None,
        created_time: Optional[int] = None,
        alternate_greetings: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        creator: Optional[str] = "",
        character_version: Optional[str] = "",
        extensions: Optional[Dict[str, Any]] = None,
        character_book: Optional[Dict[str, Any]] = None,  # Assuming it's a dict as per spec
    ):
        self.name = name
        self.summary = summary
        self.personality = personality
        self.scenario = scenario

        # Optional fields
        self.first_greeting_message = first_greeting_message
        self.greeting_message = greeting_message
        self.example_messages = example_messages
        self.image_path = image_path
        self.alternate_greetings = alternate_greetings or []
        self.tags = tags or []
        self.creator = creator
        self.character_version = character_version
        self.extensions = extensions or {}
        self.character_book = character_book  # Can be None if not provided

        # Timestamp in seconds for 'created_time'
        self.created_time = created_time or int(datetime.now().timestamp())

    @property
    def data_summary(self) -> str:
        """
        A quick summary of the character’s data as a string.
        """
        summary_str = f"Name: {self.name}\n"
        summary_str += f"Summary: {self.summary}\n"
        summary_str += f"Personality: {self.personality}\n"
        summary_str += f"Scenario: {self.scenario}\n"
        summary_str += f"First Greeting Message: {self.first_greeting_message}\n"
        summary_str += f"Greeting Message: {self.greeting_message}\n"
        summary_str += f"Example Messages: {self.example_messages}\n"
        if self.alternate_greetings:
            summary_str += "Alternate Greetings:\n" + "\n".join(self.alternate_greetings) + "\n"
        else:
            summary_str += "Alternate Greetings: None\n"

        summary_str += f"Tags: {', '.join(self.tags) if self.tags else 'None'}\n"
        summary_str += f"Creator: {self.creator}\n"
        summary_str += f"Character Version: {self.character_version}\n"
        summary_str += f"Extensions: {json.dumps(self.extensions, ensure_ascii=False)}\n"
        summary_str += f"Character Book: {json.dumps(self.character_book, ensure_ascii=False) if self.character_book else 'None'}\n"
        summary_str += f"Image Path: {self.image_path if self.image_path else 'None'}\n"
        return summary_str

    ##############################
    # Export to JSON
    ##############################

    def export_json(self, format_type: str) -> str:
        """
        Exports the character to a JSON string, per the specified format.
        e.g., format_type can be "tavernai", "sillytavern", "pygmalion", "neutral", etc.
        """
        if format_type.lower() == "neutral":
            return export_as_neutral_json(self)
        else:
            return export_as_json(self, format_type)

    def export_json_file(self, format_type: str, export_json_path: str) -> None:
        """
        Exports the character data to a JSON file at `export_json_path`.
        """
        with open(export_json_path, "w", encoding="utf-8") as f:
            f.write(self.export_json(format_type))

    ##############################
    # Export to YAML
    ##############################

    def export_yaml(self, format_type: str) -> str:
        """
        Exports the character to a YAML string, per the specified format.
        """
        if format_type.lower() == "neutral":
            return export_as_neutral_yaml(self)
        else:
            return export_as_yaml(self, format_type)

    def export_yaml_file(self, format_type: str, export_yaml_path: str) -> None:
        """
        Exports the character data to a YAML file at `export_yaml_path`.
        """
        with open(export_yaml_path, "w", encoding="utf-8") as f:
            f.write(self.export_yaml(format_type))

    ##############################
    # Export to PNG "Character Card"
    ##############################

    def export_card(self, format_type: str) -> bytes:
        """
        Returns the raw PNG bytes with a tEXt chunk named "chara" that holds
        base64-encoded JSON describing this character.
        """
        return export_as_card(self, format_type)

    def export_card_file(self, format_type: str, export_card_path: str) -> None:
        """
        Exports the character data as a .card.png file with a "chara" text chunk.
        """
        card_bytes = self.export_card(format_type)
        with open(export_card_path, "wb") as f:
            f.write(card_bytes)

    ##############################
    # "Neutral" exports
    # (or specialized for a universal format)
    ##############################

    def export_neutral_json(self) -> str:
        return export_as_neutral_json(self)

    def export_neutral_json_file(self, export_json_path: str) -> None:
        json_str = self.export_neutral_json()
        with open(export_json_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    def export_neutral_yaml(self) -> str:
        return export_as_neutral_yaml(self)

    def export_neutral_yaml_file(self, export_yaml_path: str) -> None:
        yaml_str = self.export_neutral_yaml()
        with open(export_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)

    def export_neutral_card(self) -> bytes:
        return export_as_card(self, "neutral")

    def export_neutral_card_file(self, export_card_path: str) -> None:
        card_bytes = self.export_neutral_card()
        with open(export_card_path, "wb") as f:
            f.write(card_bytes)


############################################################
# Create & Load utility functions
############################################################

def create_character(
    name: str,
    summary: str,
    personality: str,
    scenario: str,
    first_greeting_message: str,
    greeting_message: str,
    example_messages: str,
    image_path: Optional[str] = None,
    alternate_greetings: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    creator: Optional[str] = "",
    character_version: Optional[str] = "",
    extensions: Optional[Dict[str, Any]] = None,
    character_book: Optional[Dict[str, Any]] = None,
) -> CharacterClass:
    """
    Create a CharacterClass object from the supplied parameters.
    """
    return CharacterClass(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        first_greeting_message=first_greeting_message,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=image_path,
        alternate_greetings=alternate_greetings or [],
        tags=tags or [],
        creator=creator or "",
        character_version=character_version or "",
        extensions=extensions or {},
        character_book=character_book,
    )


def load_character_json(json_str: str) -> CharacterClass:
    """
    Load a character from a JSON string. This tries to unify multiple known keys:
    "char_name", "name", "char_persona", "personality", "scenario", etc.
    """
    char_data = json.loads(json_str)

    # Extract 'data' key if present
    if 'data' in char_data:
        data = char_data['data']
    else:
        data = char_data

    # Extract 'chara' and 'metadata' if present within 'data'
    chara = data.get('chara', {})
    metadata = data.get('metadata', {})

    # Ensure all fields are strings
    name = str(chara.get("char_name") or chara.get("name", ""))
    summary = str(chara.get("summary") or chara.get("description", ""))
    personality = str(chara.get("char_persona") or chara.get("personality", ""))
    scenario = str(chara.get("world_scenario") or chara.get("scenario", ""))
    first_greeting_message = str(chara.get("first_mes") or chara.get("char_greeting", ""))
    greeting_message = str(chara.get("char_greeting") or chara.get("greeting_message", ""))
    example_messages = str(chara.get("mes_example") or chara.get("example_dialogue", ""))
    alternate_greetings = chara.get("alternate_greetings", [])
    tags = chara.get("tags", [])
    creator = chara.get("creator", "")
    character_version = chara.get("character_version", "")
    extensions = chara.get("extensions", {})
    character_book = chara.get("character_book", None)

    return CharacterClass(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        first_greeting_message=first_greeting_message,
        greeting_message=greeting_message,
        example_messages=example_messages,
        alternate_greetings=alternate_greetings,
        tags=tags,
        creator=creator,
        character_version=character_version,
        extensions=extensions,
        character_book=character_book,
        created_time=metadata.get("created"),
    )


def load_character_json_file(path: str) -> CharacterClass:
    """
    Load a character from a JSON file at `path`.
    """
    with open(path, "r", encoding="utf-8") as f:
        return load_character_json(f.read())


def load_character_yaml(yaml_str: str) -> CharacterClass:
    """
    Load a character from a YAML string.
    """
    char_data = yaml.safe_load(yaml_str)

    # Extract 'data' key if present
    if 'data' in char_data:
        data = char_data['data']
    else:
        data = char_data

    # Extract 'chara' and 'metadata' if present within 'data'
    chara = data.get('chara', {})
    metadata = data.get('metadata', {})

    # Ensure all fields are strings
    name = str(chara.get("char_name") or chara.get("name", ""))
    summary = str(chara.get("summary") or chara.get("description", ""))
    personality = str(chara.get("char_persona") or chara.get("personality", ""))
    scenario = str(chara.get("world_scenario") or chara.get("scenario", ""))
    first_greeting_message = str(chara.get("first_mes") or chara.get("char_greeting", ""))
    greeting_message = str(chara.get("char_greeting") or chara.get("greeting_message", ""))
    example_messages = str(chara.get("mes_example") or chara.get("example_dialogue", ""))
    alternate_greetings = chara.get("alternate_greetings", [])
    tags = chara.get("tags", [])
    creator = chara.get("creator", "")
    character_version = chara.get("character_version", "")
    extensions = chara.get("extensions", {})
    character_book = chara.get("character_book", None)

    return CharacterClass(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        first_greeting_message=first_greeting_message,
        greeting_message=greeting_message,
        example_messages=example_messages,
        alternate_greetings=alternate_greetings,
        tags=tags,
        creator=creator,
        character_version=character_version,
        extensions=extensions,
        character_book=character_book,
        created_time=metadata.get("created"),
    )


def load_character_yaml_file(path: str) -> CharacterClass:
    """
    Load a character from a YAML file at `path`.
    """
    with open(path, "r", encoding="utf-8") as f:
        return load_character_yaml(f.read())


def load_character_card(card_bytes: bytes) -> CharacterClass:
    """
    Load a character from a PNG's 'chara' text chunk, which must contain
    base64-encoded JSON. Raises ValueError if 'chara' is missing.
    """
    image = Image.open(BytesIO(card_bytes))
    if "chara" not in image.info:
        raise ValueError("Invalid character card: 'chara' metadata not found")

    base64_str = image.info["chara"]
    logging.debug("Retrieved base64-encoded 'chara' chunk from PNG.")
    logging.debug(base64_str)

    try:
        json_str = base64.b64decode(base64_str).decode("utf-8")
        logging.debug("Decoded JSON string from base64:")
        logging.debug(json_str)
    except Exception as e:
        logging.error(f"Failed to decode base64 string: {e}")
        raise

    return load_character_json(json_str)


def load_character_card_file(path: str) -> CharacterClass:
    """
    Load a character from a PNG file on disk.
    """
    with open(path, "rb") as f:
        return load_character_card(f.read())


############################################################
# Export to JSON, YAML, PNG card
############################################################

def export_as_json(character: CharacterClass, format_type: str) -> str:
    """
    Exports the character as a JSON string in the Character Card V2 format.
    """
    current_time = int(datetime.now().timestamp())
    metadata = {
        "version": 1,
        "created": character.created_time,
        "modified": current_time,
        "source": "",  # Empty string as per spec
        "tool": {
            "name": "aicharl Python library",
            "version": "1.0.0",
            "url": "https://github.com/Hukasx0/aichar",
        },
    }

    # Filter out empty alternate greetings
    alt_greetings = [g for g in (character.alternate_greetings or []) if g.strip()]

    # Character Card V2 structure
    chara_data = {
        "name": character.name,
        "description": character.summary,
        "personality": character.personality,
        "scenario": character.scenario,
        "first_mes": character.first_greeting_message,
        "mes_example": character.example_messages,
        "creator_notes": getattr(character, 'creator_notes', ""),
        "system_prompt": getattr(character, 'system_prompt', ""),
        "post_history_instructions": getattr(character, 'post_history_instructions', ""),
        "alternate_greetings": alt_greetings,
        "tags": character.tags or [],
        "creator": character.creator or "",
        "character_version": character.character_version or "",
        "extensions": character.extensions or {},
        "character_book": character.character_book or None,  # Optional field
    }

    # Wrap in the V2 spec structure
    export_data = {
        "spec": "chara_card_v2",  # Required for V2
        "spec_version": "2.0",    # Required for V2
        "data": chara_data,       # Directly include character fields here
        "metadata": metadata,
    }

    # Convert to JSON string
    json_output = json.dumps(export_data, ensure_ascii=False, indent=4)
    logging.debug("Generated JSON for export_as_json:")
    logging.debug(json_output)
    return json_output



def export_as_neutral_json(character: CharacterClass) -> str:
    """
    A 'neutral' JSON format that merges all known fields into a single set.
    This is often acceptable by multiple frontends.
    """
    current_time = int(time.time())
    metadata = {
        "version": 1,
        "created": character.created_time,
        "modified": current_time,
        "source": "",  # Changed from None to empty string
        "tool": {
            "name": "aicharl Python library",
            "version": "1.0.0",
            "url": "https://github.com/Hukasx0/aichar",
        },
    }

    alt_greetings = [g.strip() for g in (character.alternate_greetings or []) if g.strip()]

    chara_data = {
        "name": character.name,  # Changed from "char_name" to "name" as per V2 spec
        "description": character.summary,  # Changed from "summary" to "description"
        "personality": character.personality,
        "scenario": character.scenario,
        "first_mes": character.first_greeting_message.strip(),
        "mes_example": character.example_messages.strip(),
        "creator_notes": getattr(character, 'creator_notes', ""),
        "system_prompt": getattr(character, 'system_prompt', ""),
        "post_history_instructions": getattr(character, 'post_history_instructions', ""),
        "alternate_greetings": alt_greetings,
        "tags": character.tags,  # Added tags
        "creator": character.creator,  # Added creator
        "character_version": character.character_version,  # Added character_version
        "extensions": character.extensions,  # Added extensions
        "character_book": character.character_book  # Added character_book (optional)
    }

    export_data = {
        "spec": "chara_card_v2",  # Added spec field
        "spec_version": "2.0",     # Added spec_version field
        "data": chara_data,        # Directly include character fields here
        "metadata": metadata,      # Keep metadata at top level
    }

    json_output = json.dumps(export_data, ensure_ascii=False, indent=4)
    logging.debug("Generated JSON for export_as_neutral_json:")
    logging.debug(json_output)
    return json_output



def export_as_yaml(character: CharacterClass, format_type: str) -> str:
    """
    Same as `export_as_json`, but converted to YAML.
    """
    json_str = export_as_json(character, format_type)
    yaml_output = yaml.dump(json.loads(json_str), default_flow_style=False, allow_unicode=True)
    logging.debug("Converted JSON to YAML:")
    logging.debug(yaml_output)
    return yaml_output


def export_as_neutral_yaml(character: CharacterClass) -> str:
    """
    Same as `export_as_neutral_json`, but YAML.
    """
    json_str = export_as_neutral_json(character)
    yaml_output = yaml.safe_dump(json.loads(json_str), default_flow_style=False, allow_unicode=True)
    logging.debug("Converted neutral JSON to YAML:")
    logging.debug(yaml_output)
    return yaml_output


def export_as_card(character: CharacterClass, format_type: str) -> bytes:
    """
    Exports a PNG with a tEXt chunk named "chara" containing base64-encoded JSON data.
    The JSON data is properly structured with "data" as a top-level key.
    """
    if not character.image_path:
        raise ValueError("Cannot export as card; 'image_path' is missing.")

    try:
        # 1) Load the user-supplied PNG
        image = Image.open(character.image_path)
        logging.debug(f"Loaded image from {character.image_path}")

        # 2) Get the JSON string with the correct structure
        if format_type.lower() == "neutral":
            export_json_str = export_as_neutral_json(character)
        else:
            export_json_str = export_as_json(character, format_type)
        logging.debug("Exported JSON string:")
        logging.debug(export_json_str)

        # 3) Validate JSON structure
        try:
            json_obj = json.loads(export_json_str)
            logging.debug("JSON structure is valid.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            raise

        # 4) Base64-encode the entire JSON string
        base64_str = base64.b64encode(export_json_str.encode("utf-8")).decode("utf-8")
        logging.debug("Base64-encoded JSON string:")
        logging.debug(base64_str)

        # 5) Store it in the "chara" text chunk
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("chara", base64_str)
        logging.debug("Added 'chara' text chunk to PNG.")

        # 6) Write to a buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG", pnginfo=pnginfo)
        buffer.seek(0)
        logging.debug("Saved image to buffer.")

        return buffer.getvalue()

    except Exception as e:
        logging.error(f"Error exporting card: {e}")
        raise


############################################################
# License
############################################################

def license() -> str:
    """
    MIT License
    """
    return """
    MIT License

    Copyright (c) 2023-2025 Hubert Kasperek

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
