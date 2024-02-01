import os
import aichar
import requests
from bs4 import BeautifulSoup
from diffusers import DiffusionPipeline
import torch
import gradio as gr
from PIL import Image
import re

llm = None
sd = None
safety_checker_sd = None

folder_path = "models"
global_url = ""


def process_url(url):
    global global_url
    global_url = url.rstrip("/") + "/v1/completions"  # Append '/v1/completions' to the URL
    return f"URL Set: {global_url}"  # Return the modified URL


# Function to process the uploaded image
from PIL import Image
import numpy as np


def process_uploaded_image(uploaded_img):
    global processed_image_path  # Access the global variable

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(np.uint8(uploaded_img)).convert('RGB')

    # Define the path for the uploaded image
    character_name = "uploaded_character"
    os.makedirs(f"characters/{character_name}", exist_ok=True)
    image_path = f"characters/{character_name}/{character_name}.png"

    # Save the image
    pil_image.save(image_path)

    # Update the global variable with the image path
    processed_image_path = image_path

    print("Uploaded image saved at:", image_path)
    return pil_image


def send_message(prompt):
    global global_url
    if not global_url:
        return "Error: URL not set."
    request = {
        'prompt': prompt,
        'max_new_tokens': 1024,
        "max_tokens": 8192,
        'do_sample': True,
        'temperature': 1.1,
        'top_p': 0.95,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'add_bos_token': True,
        'truncation_length': 8192,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stop': [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
        ],
        'stopping_strings': [
            "/s",
            "</s>",
            "<s>",
            "<|system|>",
            "<|assistant|>",
            "<|user|>",
            "<|char|>",
        ]
    }

    try:
        response = requests.post(global_url, json=request)
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '')
        return result
    except requests.RequestException as e:
        return f"Error sending request: {e}"


def load_models():
    global sd
    sd = DiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=False,
    )
    if torch.cuda.is_available():
        sd.to("cuda")
        print("Loading Stable Diffusion to GPU...")
    else:
        print("Loading Stable Diffusion to CPU...")
    global llm
    gpu_layers = 0
    if torch.cuda.is_available():
        gpu_layers = 110
        print("Loading LLM to GPU...")
    else:
        print("Loading LLM to CPU...")


load_models()

import requests
from bs4 import BeautifulSoup


def generate_character_name(topic, gender):
    example_dialogue = """
<|system|>
You are a text generation tool, you should always just return the name of the character and nothing else, you should not ask any questions.
You only answer by giving the name of the character, you do not describe it, you do not mention anything about it. You can't write anything other than the character's name.
</s>
<|user|> Generate a random character name. Topic: business. Gender: male </s>
<|assistant|> Jamie Hale </s>
<|user|> Generate a random character name. Topic: fantasy </s>
<|assistant|> Eldric </s>
<|user|> Generate a random character name. Topic: anime. Gender: female </s>
<|assistant|> Tatsukaga Yamari </s>
<|user|> Generate a random character name. Topic: {{user}}'s pet cat. </s>
<|assistant|> mr. Fluffy </s>
    """  # nopep8
    gender = input_none(gender)
    output = send_message(
        example_dialogue
        + "\n<|user|> Generate a random character name. "
        + f"Topic: {topic}. "
        + f"{'Character gender: ' + gender + '.' if gender else ''} "
        + "</s>\n<|assistant|> "
    )
    output = re.sub(r"[^a-zA-Z0-9_ -]", "", output).strip()
    print(output)
    return output


'''    <|system|>: You are a text generation tool. Describe the character in a very simple and understandable way, you can just list some characteristics, you do not need to write a professional characterization of the character. Describe: age, height, personality traits, appearance, clothing, what the character likes, what the character does not like.
    You must not write any summaries, overalls, endings or character evaluations at the end, you just have to return the character's personality and physical traits.
    Don't ask any questions, don't inquire about anything.
    The topic given by the user is to serve as a background to the character, not as the main theme of your answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style. If user gives as the topic eg. 'noir style detective', you do not return things like:
    'Character is a noir style detective', you just describe it so that the character fits that theme. Use simple and understandable English, use simple and colloquial terms.
    You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.
    Should include in its description who the character is - for example, a human mage, an elf archer, a shiba dog.
    Should be in the same form as the previous answers.
    You must include character traits, physical and character. You can't add anything else.
    </s>
'''


def generate_character_summary(character_name, topic, table_data, gender, content=None):
    content_clause = f"using {content}" if content else ""

    example_dialogue = """

    <|user|>: Please generate a detailed character profile for John Everbrave based on the following attributes. Use lower case and separate attributes with commas. Fill in only the relevant attributes. Some attributes are optional or redundant - choose appropriately. Character gender: male, Tailor the character to the theme of history but don't specify what topic it is, and don't describe the topic itself. </s>

    <|assistant|>:
    Name: John Everbrave
    AKA: The Lionheart
    Type: character
    Setting: Medieval fantasy world
    Species: Human
    Gender: Male
    Age: 32
    Height: 6'2"
    Weight: 190 lbs
    Appearance: Broad-shouldered, muscular, with piercing blue eyes and dark brown hair
    Clothing: Wears a suit of shining armor, adorned with the crest of his kingdom
    Attire: A red cape that signifies his status as a knight
    Personality: Brave, righteous, compassionate
    Mind: Strategic thinker
    Mental: Strong-willed
    Likes: Justice, peace, helping others
    Dislikes: Injustice, tyranny
    Sexuality: Heterosexual
    Speech: Commanding yet kind, with a firm tone
    Voice: Deep and resonant
    Abilities (optional): Skilled swordsman, excellent horseman
    Skills (optional): Leadership, combat strategy
    Quote: "For the honor of Eldoria!"
    Affiliation: Kingdom of Eldoria
    Occupation: Knight
    Reputation: Known as a hero across the land
    Secret: Heir to a mysterious ancient power
    Family: Son of Duke Everbrave
    Allies: The Eldorian people, the Wizard Merlyn
    Enemies: The Dark Sorcerer, Orc hordes
    Background: Raised as a noble, trained as a knight, destined to protect the realm
    Description: John Everbrave is a symbol of hope and courage in a land beset by darkness.
    Attributes: Loyal, honorable, physically strong
    </s>
"""  # nopep8
    gender = input_none(gender)
    output = send_message(
        example_dialogue
        + "\n<|user|> Create a longer description for a character named "
        + f"{character_name}. "
        + f"{'Character gender: ' + gender + '.' if gender else ''} "
        + f"this is the character's data: {content_clause} and {table_data}"
        + "Describe their appearance, distinctive features, and looks. "
        + f"Tailor the character to the theme of {topic} but don't "
        + "specify what topic it is, and don't describe the topic itself. "
        + "You are to write a brief description of the character. You must "
        + "include character traits, physical and character. You can't add "
        + "anything else. You must not write any summaries, conclusions or "
        + "endings. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_character_personality(
        character_name,
        character_summary,
        topic
):
    example_dialogue = """
<|system|>
You are a text generation tool. Describe the character personality in a very simple and understandable way.
You can simply list the most suitable character traits for a given character, the user-designated character description as well as the theme can help you in matching personality traits.
Don't ask any questions, don't inquire about anything.
You must describe the character in the present tense, even if it is a historical figure who is no longer alive. you can't use future tense or past tense to describe a character.
Don't write any summaries, endings or character evaluations at the end, you just have to return the character's personality traits. Use simple and understandable English, use simple and colloquial terms.
You are not supposed to write characterization of the character, you don't have to form terms whether the character is good or bad, only you are supposed to write out the character traits of that character, nothing more.
You must return character traits in your answers, you can not describe the appearance, clothing, or who the character is, only character traits.
Your answer should be in the same form as the previous answers.
</s>
<|user|> Describe the personality of Jamie Hale. Their characteristics Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him </s>
<|assistant|> Jamie Hale is calm, stoic, focused, intelligent, sensitive to art, discerning, focused, motivated, knowledgeable about business, knowledgeable about new business technologies, enjoys reading business and science books </s>
<|user|> Describe the personality of Mr Fluffy. Their characteristics  Mr fluffy is {{user}}'s cat who is very fat and fluffy, he has black and white colored fur, this cat is 3 years old, he loves special expensive cat food and lying on {{user}}'s lap while he does his homework. Mr. Fluffy can speak human language, he is a cat who talks a lot about philosophy and expresses himself in a very sarcastic way </s>
<|assistant|> Mr Fluffy is small, calm, lazy, mischievous cat, speaks in a very philosophical manner and is very sarcastic in his statements, very intelligent for a cat and even for a human, has a vast amount of knowledge about philosophy and the world </s>
"""  # nopep8
    output = send_message(
        example_dialogue
        + f"\n<|user|> Describe the personality of {character_name}. "
        + f"Their characteristic {character_summary}\nDescribe them "
        + "in a way that allows the reader to better understand their "
        + "character. Make this character unique and tailor them to "
        + f"the theme of {topic} but don't specify what topic it is, "
        + "and don't describe the topic itself. You are to write out "
        + "character traits separated by commas, you must not write "
        + "any summaries, conclusions or endings. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_character_scenario(
        character_summary,
        character_personality,
        topic
):
    example_dialogue = """
<|system|>
You are a text generation tool.
The topic given by the user is to serve as a background to the character, not as the main theme of your answer.
Use simple and understandable English, use simple and colloquial terms.
You must include {{user}} and {{char}} in your response.
Your answer must be very simple and tailored to the character, character traits and theme.
Your answer must not contain any dialogues.
Instead of using the character's name you must use {{char}}.
Your answer should be in the same form as the previous answers.
Your answer must be short, maximum 5 sentences.
You can not describe the character, but you have to describe the scenario and actions.
</s>
<|user|> Write a simple and undemanding introduction to the story, in which the main characters will be {{user}} and {{char}}, do not develop the story, write only the introduction. {{char}} characteristics: Tatsukaga Yamari is an 23 year old anime girl, who loves books and coffee. Make this character unique and tailor them to the theme of anime, but don't specify what topic it is, and don't describe the topic itself. Your response must end when {{user}} and {{char}} interact. </s>
<|assistant|> When {{user}} found a magic stone in the forest, he moved to the magical world, where he meets {{char}}, who looks at him in disbelief, but after a while comes over to greet him. </s>
"""  # nopep8
    output = send_message(
        example_dialogue
        + f"\n<|user|> Write a scenario for chat roleplay "
        + "to serve as a simple storyline to start chat "
        + "roleplay by {{char}} and {{user}}. {{char}} "
        + f"characteristics: {character_summary}. "
        + f"{character_personality}. Make this character unique "
        + f"and tailor them to the theme of {topic} but don't "
        + "specify what topic it is, and don't describe the topic "
        + "itself. Your answer must not contain any dialogues. "
        + "Your response must end when {{user}} and {{char}} interact. "
        + "</s>\n<|assistant|> "
    )
    print(output)
    return output


def generate_character_greeting_message(
        character_name, character_summary, character_personality, topic
):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear. You play the provided character and you write a message that you would start a chat roleplay with {{user}}. The form of your answer should be similar to previous answers.
The topic given by the user is only to be an aid in selecting the style of the answer, not the main purpose of the answer, e.g. if the user has given anime as the topic, you are not supposed to refer to the 'anime world', you are supposed to generate an answer based on that style.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
</s>
<|user|> Create the first message that the character Tatsukaga Yamari, whose personality is: a vibrant tapestry of enthusiasm, curiosity, and whimsy. She approaches life with boundless energy and a spirit of adventure, always ready to embrace new experiences and challenges. Yamari is a compassionate and 
caring friend, offering solace and support to those in need, and her infectious laughter brightens the lives of those around her. Her unwavering loyalty and belief in the power of friendship define her character, making her a heartwarming presence in the story she inhabits. Underneath her playful exterior lies a wellspring of inner strength, as she harnesses incredible magical abilities to overcome adversity and protect her loved ones.\n greets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Tatsukaga Yamari's eyes light up with curiosity and wonder as she warmly greets you*, {{user}}! *With a bright and cheerful smile, she exclaims* Hello there, dear friend! It's an absolute delight to meet you in this whimsical world of imagination. I hope you're ready for an enchanting adventure, full of surprises and magic. What brings you to our vibrant anime-inspired realm today? </s>
<|user|> Create the first message that the character Jamie Hale, whose personality is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional.\nJamie Hale's personality is characterized by his unwavering determination and sharp intellect. He exudes confidence and charisma, drawing people to him with his commanding presence and air of authority. He is a natural leader, known for his shrewd 
decision-making in the business world, and he possesses an insatiable thirst for success. Despite his professional achievements, he values his family and close friends, maintaining a strong work-life balance, and he has a penchant for enjoying the finer things in life, such as upscale dining and the arts.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Jamie Hale extends a firm, yet friendly, handshake as he greets you*, {{user}}. *With a confident smile, he says* Greetings, my friend. It's a pleasure to make your acquaintance. In the world of business and beyond, it's all about seizing opportunities and making every moment count. What can I assist you with today, or perhaps, share a bit of wisdom about navigating the path to success? </s>
<|user|> Create the first message that the character Eldric, whose personality is Eldric is a strikingly elegant elf who has honed his skills as an archer and possesses a deep connection to the mystical arts. Standing at a lithe and graceful 6 feet, his elven heritage is evident in his pointed ears, ethereal features, and eyes that shimmer with an otherworldly wisdom.\nEldric possesses a serene and contemplative nature, reflecting the wisdom of his elven heritage. He is deeply connected to the natural world, showing a profound respect for the environment and its creatures. Despite his formidable combat 
abilities, he prefers peaceful solutions and seeks to maintain harmony in his woodland domain.\ngreets the user we are addressing as {{user}}. Make this character unique and tailor them to the theme of fantasy but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> *Eldric, the elegant elf, approaches you with a serene and contemplative air. His shimmering eyes, filled with ancient wisdom, meet yours as he offers a soft and respectful greeting* Greetings, {{user}}. It is an honor to welcome you to our enchanted woodland realm. I am Eldric, guardian of this forest, and I can sense that you bring a unique energy with you. How may I assist you in your journey through the wonders of the natural world or share the mysteries of our elven heritage with you today? </s>
"""  # nopep8
    output = send_message(
        example_dialogue
        + "\n<|user|> Create the first message that the character "
        + f"{character_name}, whose personality is "
        + f"{character_summary}\n{character_personality}\n "
        + "greets the user we are addressing as {{user}}. "
        + "Make this character unique and tailor them to the theme "
        + f"of {topic} but don't specify what topic it is, "
        + "and don't describe the topic itself. You must match the "
        + "speaking style to the character, if the character is "
        + "childish then speak in a childish way, if the character "
        + "is serious, philosophical then speak in a serious and "
        + "philosophical way, and so on. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_example_messages(
        character_name, character_summary, character_personality, topic
):
    example_dialogue = """
<|system|>
You are a text generation tool, you are supposed to generate answers so that they are simple and clear.
Your answer should be a dialog between {{user}} and {{char}}, where {{char}} is the specified character. The dialogue must be several messages taken from the roleplay chat between the user and the character.
Only respond in {{user}} or {{char}} messages. The form of your answer should be similar to previous answers.
You must match the speaking style to the character, if the character is childish then speak in a childish way, if the character is serious, philosophical then speak in a serious and philosophical way and so on.
If the character is shy, then needs to speak little and quietly, if the character is aggressive then needs to shout and speak a lot and aggressively, if the character is sad then needs to be thoughtful and quiet, and so on.
Dialog of {{user}} and {{char}} must be appropriate to their character traits and the way they speak.
Instead of the character's name you must use {{char}}.
</s>
<|user|> Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Jamie Hale. Jamie Hale characteristics: Jamie Hale is an adult, intelligent well-known and respected businessman. Make this character unique and tailor them to the theme of business but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> {{user}}: Good afternoon, Mr. {{char}}. I've heard so much about your success in the corporate world. It's an honor to meet you.
{{char}}: *{{char}} gives a warm smile and extends his hand for a handshake.* The pleasure is mine, {{user}}. Your reputation precedes you. Let's make this venture a success together.
{{user}}: *Shakes {{char}}'s hand with a firm grip.* I look forward to it.
{{char}}: *As they release the handshake, Jamie leans in, his eyes sharp with interest.* Impressive. Tell me more about your innovations and how they align with our goals. </s>
<|user|> Create a dialogue between {{user}} and {{char}}, they should have an interesting and engaging conversation, with some element of interaction like a handshake, movement, or playful gesture. Make it sound natural and dynamic. {{char}} is Tatsukaga Yamari. Tatsukaga Yamari characteristics: Tatsukaga Yamari is an anime girl, living in a magical world and solving problems. Make this character unique and tailor them to the theme of anime but don't specify what topic it is, and don't describe the topic itself </s>
<|assistant|> {{user}}: {{char}}, this forest is absolutely enchanting. What's the plan for our adventure today?
{{char}}: *{{char}} grabs {{user}}'s hand and playfully twirls them around before letting go.* Well, we're off to the Crystal Caves to retrieve the lost Amethyst Shard. It's a treacherous journey, but I believe in us.
{{user}}: *Nods with determination.* I have no doubt we can do it. With your magic and our unwavering friendship, there's nothing we can't accomplish.
{{char}}: *{{char}} moves closer, her eyes shining with trust and camaraderie.* That's the spirit, {{user}}! Let's embark on this epic quest and make the Crystal Caves ours! </s>
"""  # nopep8
    output = send_message(
        example_dialogue
        + f"\n<|user|> Create a dialogue between {{user}} and {{char}}, "
        + "they should have an interesting and engaging conversation, "
        + "with some element of interaction like a handshake, movement, "
        + "or playful gesture. Make it sound natural and dynamic. "
        + f"{{char}} is {character_name}. {character_name} characteristics: "
        + f"{character_summary}. {character_personality}. Make this "
        + f"character unique and tailor them to the theme of {topic} but "
        + "don't specify what topic it is, and don't describe the "
        + "topic itself. You must match the speaking style to the character, "
        + "if the character is childish then speak in a childish way, if the "
        + "character is serious, philosophical then speak in a serious and "
        + "philosophical way and so on. </s>\n<|assistant|> "
    ).strip()
    print(output)
    return output


def generate_character_avatar(
        character_name,
        character_summary,
        topic,
        negative_prompt,
        avatar_prompt,
        nsfw_filter,
):
    example_dialogue = """
<|system|>
You are a text generation tool, in the response you are supposed to give only descriptions of the appearance, what the character looks like, describe the character simply and unambiguously
</s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business </s> 
<|assistant|> male, realistic, human, Confident and commanding presence, Polished and professional appearance, tailored suit, Well-maintained physique, Dark well-groomed hair, Neatly trimmed beard, blue eyes </s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, 
cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime </s>
<|assistant|> female, anime, Petite and delicate frame, Raven-black hair flowing down to her waist, Striking purple ribbon in her hair, Large and expressive amethyst-colored eyes, Colorful and eclectic outfit, oversized bows, cat-eared headbands, mismatched socks </s>
<|user|> create a prompt that lists the appearance characteristics of a character whose summary is Name: suzie Summary: Topic: none Gender: none</s>
<|assistant|> 1girl, improvised tag, </s>
"""  # nopep8
    sd_prompt = (
            input_none(avatar_prompt)
            or send_message(
        example_dialogue
        + "\n<|user|> create a prompt that lists the appearance "
        + "characteristics of a character whose summary is "
        + "if lack of info, generate something based on available info."
        + f"{character_summary}. Topic: {topic}</s>\n<|assistant|> "
    ).strip()
    )
    print(sd_prompt)
    sd_filter(nsfw_filter)
    return image_generate(character_name,
                          sd_prompt,
                          input_none(negative_prompt),
                          )


def image_generate(character_name, prompt, negative_prompt):
    prompt = "absurdres, full hd, 8k, high quality, " + prompt
    default_negative_prompt = (
            "worst quality, normal quality, low quality, low res, blurry, "
            + "text, watermark, logo, banner, extra digits, cropped, "
            + "jpeg artifacts, signature, username, error, sketch, "
            + "duplicate, ugly, monochrome, horror, geometry, "
            + "mutation, disgusting, "
            + "bad anatomy, bad hands, three hands, three legs, "
            + "bad arms, missing legs, missing arms, poorly drawn face, "
            + " bad face, fused face, cloned face, worst face, "
            + "three crus, extra crus, fused crus, worst feet, "
            + "three feet, fused feet, fused thigh, three thigh, "
            + "fused thigh, extra thigh, worst thigh, missing fingers, "
            + "extra fingers, ugly fingers, long fingers, horn, "
            + "extra eyes, huge eyes, 2girl, amputation, disconnected limbs"
    )
    negative_prompt = default_negative_prompt + (negative_prompt or "")

    generated_image = sd(prompt, negative_prompt=negative_prompt).images[0]

    character_name = character_name.replace(" ", "_")
    os.makedirs(f"characters/{character_name}", exist_ok=True)

    card_path = f"characters/{character_name}/{character_name}.png"

    generated_image.save(card_path)
    print("Generated character avatar")
    return generated_image


def sd_filter(enable):
    if enable:
        sd.safety_checker = safety_checker_sd
        sd.requires_safety_checker = True
    else:
        sd.safety_checker = None
        sd.requires_safety_checker = False


def input_none(text):
    user_input = text
    if user_input == "":
        return None
    else:
        return user_input


"""## Start WebUI"""


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

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    if processed_image_path is not None:
        # If an image has been processed, use it
        image_path = processed_image_path
    else:
        # If no image has been processed, use a default or placeholder image
        # e.g., image_path = "path/to/default/image.png"
        image_path = None  # Or set a default image path

    # Create the character with the appropriate image
    character = aichar.create_character(
        name=name,
        summary=summary,
        personality=personality,
        scenario=scenario,
        greeting_message=greeting_message,
        example_messages=example_messages,
        image_path=image_path  # Use the processed or default image
    )

    # Export the character card
    card_path = f"{base_path}{character_name}.card.png"
    character.export_neutral_card_file(card_path)
    return Image.open(card_path)


def scrape_fandom_character(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the title of the page
    title = soup.find('h1', {'class': 'page-header__title'}).get_text(strip=True)

    # Extract structured data
    structured_content = {
        "headers": [],
        "paragraphs": [],
        "tables": []
    }

    # Headers
    for header in soup.find_all(['h2', 'h3']):
        # Skip edit sections
        if 'mw-editsection' not in header.get('class', []):
            structured_content["headers"].append(header.get_text(strip=True))

    # Paragraphs
    for para in soup.find_all('p'):
        structured_content["paragraphs"].append(para.get_text(strip=True))

    # Tables
    for table in soup.find_all('table'):
        table_data = []
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['th', 'td']):
                row_data.append(cell.get_text(strip=True))
            table_data.append(row_data)
        structured_content["tables"].append(table_data)

    return title, structured_content


def set_character_data(url, table_data, gender):
    gender = input_none(gender)
    # Scrape the character data from the URL
    name, structured_content = scrape_fandom_character(url)

    # Aggregate the scraped data into a single string for context
    scraped_data = " ".join(structured_content["paragraphs"] + structured_content["headers"])
    scraped_tables = " ".join([" ".join(row) for table in structured_content["tables"] for row in table])

    # AI-based content generation for each character attribute
    summary = generate_character_summary(name, scraped_data, table_data, gender)
    '''personality = generate_character_personality(name, scraped_data, scraped_tables)
    scenario = generate_character_scenario(name, scraped_data, scraped_tables)
    greeting_message = generate_character_greeting_message(name, scraped_data)
    example_messages = generate_example_messages(name, scraped_data, scraped_tables)'''
    personality = generate_character_personality(name, summary, "")
    scenario = generate_character_scenario(summary, personality, "")
    greeting_message = generate_character_greeting_message(name, summary, personality, "")
    example_messages = generate_example_messages(name, summary, personality, "")
    return name, summary, personality, scenario, greeting_message, example_messages
    # Return the generated character data in the required format


'''return {
        "name": name,
        "summary": summary,
        "personality": personality,
        "scenario": scenario,
        "greeting_message": greeting_message,
        "example_messages": example_messages
    }'''

with gr.Blocks() as webui:
    gr.Markdown("# Character Factory WebUI")
    gr.Markdown("## Model: Zephyr 7b Beta")
    with gr.Row():
        url_input = gr.Textbox(label="Enter URL", value="http://127.0.0.1:5000")
        submit_button = gr.Button("Set URL")
    output = gr.Textbox(label="URL Status")

    submit_button.click(
        process_url, inputs=url_input, outputs=output
    )
    with gr.Tab("Edit character"):
        gr.Markdown(
            "## Protip: If you want to generate the entire character using LLM and Stable Diffusion, start from the top to bottom"
            # nopep8
        )
        topic = gr.Textbox(
            placeholder="Topic: The topic for character generation (e.g., Fantasy, Anime, etc.)",  # nopep8
            label="topic",
        )
        gender = gr.Textbox(
            placeholder="Gender: Gender of the character", label="gender"
        )
        with gr.Column():
            with gr.Row():
                name = gr.Textbox(placeholder="character name", label="name")
                name_button = gr.Button("Generate character name with LLM")
                name_button.click(
                    generate_character_name,
                    inputs=[topic, gender],
                    outputs=name
                )
            with gr.Row():
                summary = gr.Textbox(
                    placeholder="character summary",
                    label="summary"
                )
                summary_button = gr.Button("Generate character summary with LLM")  # nopep8
                summary_button.click(
                    generate_character_summary,
                    inputs=[name, topic, gender],
                    outputs=summary,
                )
            with gr.Row():
                personality = gr.Textbox(
                    placeholder="character personality", label="personality"
                )
                personality_button = gr.Button(
                    "Generate character personality with LLM"
                )
                personality_button.click(
                    generate_character_personality,
                    inputs=[name, summary, topic],
                    outputs=personality,
                )
            with gr.Row():
                scenario = gr.Textbox(
                    placeholder="character scenario",
                    label="scenario"
                )
                scenario_button = gr.Button("Generate character scenario with LLM")  # nopep8
                scenario_button.click(
                    generate_character_scenario,
                    inputs=[summary, personality, topic],
                    outputs=scenario,
                )
            with gr.Row():
                greeting_message = gr.Textbox(
                    placeholder="character greeting message",
                    label="greeting message"
                )
                greeting_message_button = gr.Button(
                    "Generate character greeting message with LLM"
                )
                greeting_message_button.click(
                    generate_character_greeting_message,
                    inputs=[name, summary, personality, topic],
                    outputs=greeting_message,
                )
            with gr.Row():
                example_messages = gr.Textbox(
                    placeholder="character example messages",
                    label="example messages"
                )
                example_messages_button = gr.Button(
                    "Generate character example messages with LLM"
                )
                example_messages_button.click(
                    generate_example_messages,
                    inputs=[name, summary, personality, topic],
                    outputs=example_messages,
                )
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(interactive=True, label="Character Image", width=512, height=512)
                    # Button to process the uploaded image
                    process_image_button = gr.Button("Process Uploaded Image")

                    # Function to handle the uploaded image
                process_image_button.click(
                    process_uploaded_image,  # Your function to handle the image
                    inputs=[image_input],
                    outputs=[image_input]  # You can update the same image display with the processed image
                )
                with gr.Column():
                    negative_prompt = gr.Textbox(
                        placeholder="negative prompt for stable diffusion (optional)",  # nopep8
                        label="negative prompt",
                    )
                    avatar_prompt = gr.Textbox(
                        placeholder="prompt for generating character avatar (If not provided, LLM will generate prompt from character description)",
                        # nopep8
                        label="stable diffusion prompt",
                    )
                    avatar_button = gr.Button(
                        "Generate avatar with stable diffusion (set character name first)"  # nopep8
                    )
                    potential_nsfw_checkbox = gr.Checkbox(
                        label="Block potential NSFW image (Upon detection of this content, a black image will be returned)",
                        # nopep8
                        value=True,
                        interactive=True,
                    )
                    avatar_button.click(
                        generate_character_avatar,
                        inputs=[
                            name,
                            summary,
                            topic,
                            negative_prompt,
                            avatar_prompt,
                            potential_nsfw_checkbox,
                        ],
                        outputs=image_input,
                    )
    with gr.Tab("Fandom Import"):
        with gr.Column():
            fandom_url_input = gr.Textbox(label="Enter Fandom URL")
            fandom_import_button = gr.Button("Import from Fandom")
            fandom_import_button.click(
                set_character_data,
                inputs=fandom_url_input,
                outputs=[
                    name, summary, personality, scenario, greeting_message, example_messages
                ]
            )
    with gr.Tab("Import character"):
        with gr.Column():
            with gr.Row():
                import_card_input = gr.File(
                    label="Upload character card file", file_types=[".png"]
                )
                import_json_input = gr.File(
                    label="Upload JSON file", file_types=[".json"]
                )
            with gr.Row():
                import_card_button = gr.Button("Import character from character card")  # nopep8
                import_json_button = gr.Button("Import character from json")

            import_card_button.click(
                import_character_card,
                inputs=[import_card_input],
                outputs=[
                    name,
                    summary,
                    personality,
                    scenario,
                    greeting_message,
                    example_messages,
                ],
            )
            import_json_button.click(
                import_character_json,
                inputs=[import_json_input],
                outputs=[
                    name,
                    summary,
                    personality,
                    scenario,
                    greeting_message,
                    example_messages,
                ],
            )
    with gr.Tab("Export character"):
        with gr.Column():
            with gr.Row():
                export_image = gr.Image(width=512, height=512)
                export_json_textbox = gr.JSON()

            with gr.Row():
                export_card_button = gr.Button("Export as character card")
                export_json_button = gr.Button("Export as JSON")

                export_card_button.click(
                    export_character_card,
                    inputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                    outputs=export_image,
                )
                export_json_button.click(
                    export_as_json,
                    inputs=[
                        name,
                        summary,
                        personality,
                        scenario,
                        greeting_message,
                        example_messages,
                    ],
                    outputs=export_json_textbox,
                )

safety_checker_sd = sd.safety_checker

webui.launch(debug=True)
