# image_generation.py

from utils import sd, send_message, input_none
import os
from PIL import Image
import numpy as np


def generate_character_avatar(
        character_name,
        character_summary,
        topic,
        negative_prompt,
        avatar_prompt,
        nsfw_filter,
        gender
):
    example_dialogue = """
<|system|>
You are a text generation tool, in the response you are supposed to give only descriptions of the appearance, what the character looks like, describe the character simply and unambiguously
Danbooru tags are descriptive labels used on the Danbooru image board to categorize and describe images, especially in anime and manga art. They cover a wide range of specifics such as character features, clothing styles, settings, and themes. These tags help in organizing and navigating the large volume of artwork and are also useful in guiding AI models like Stable Diffusion to generate specific images.
A brief example of Danbooru tags for an anime character might look like this:

Appearance: blue_eyes, short_hair, smiling
Clothing: school_uniform, necktie
Setting: classroom, daylight
In this example, the tags precisely describe the character's appearance (blue eyes, short hair, smiling), their clothing (school uniform with a necktie), and the setting of the image (classroom during daylight). 
</s>
<|user|>: create a prompt that lists the appearance characteristics of a character whose summary is Jamie Hale is a savvy and accomplished businessman who has carved a name for himself in the world of corporate success. With his sharp mind, impeccable sense of style, and unwavering determination, he has risen to the top of the business world. Jamie stands at 6 feet tall with a confident and commanding presence. He exudes charisma and carries himself with an air of authority that draws people to him.
Jamie's appearance is always polished and professional. He is often seen in tailored suits that accentuate his well-maintained physique. His dark, well-groomed hair and neatly trimmed beard add to his refined image. His piercing blue eyes exude a sense of intense focus and ambition. Topic: business </s> 
<|assistant|>: 1male, realistic, human, confident, commanding_presence, professional_attire, suit, fit_physique, dark_hair, well_groomed_hair, beard, neatly_trimmed_beard, blue_eyes
<|user|>: create a prompt that lists the appearance characteristics of a character whose summary is Yamari stands at a petite, delicate frame with a cascade of raven-black hair flowing down to her waist. A striking purple ribbon adorns her hair, adding an elegant touch to her appearance. Her eyes, large and expressive, are the color of deep amethyst, reflecting a kaleidoscope of emotions and sparkling with curiosity and wonder.
Yamari's wardrobe is a colorful and eclectic mix, mirroring her ever-changing moods and the whimsy of her adventures. She often sports a schoolgirl uniform, a cute kimono, or an array of anime-inspired outfits, each tailored to suit the theme of her current escapade. Accessories, such as oversized bows, 
cat-eared headbands, or a pair of mismatched socks, contribute to her quirky and endearing charm. Topic: anime </s>
<|assistant|>: 1girl, anime, petite, delicate_frame, long_hair, black_hair, waist_length_hair, hair_ribbon, purple_ribbon, purple_eyes, oversized_bow, cat_ears, mismatched_socks </s>
"""  # nopep8

    # Detect if "anime" is in the character summary or topic and adjust the prompt
    anime_specific_tag = "anime, 2d, " if 'anime' in character_summary.lower() or 'anime' in topic.lower() else "realistic, 3d, "

    raw_sd_prompt = (
            input_none(avatar_prompt)
            or send_message(
        example_dialogue
        + "\n<|user|>: create a prompt that lists the appearance "
        + "characteristics of a character whose gender is "
        + f"Gender: {gender}, if female = 1girl, if male = 1boy. "
        + f" and whose summary is:"
        + f"{character_summary}. Topic: {topic}</s>\n<|assistant|>: "
    ).strip()
    )

    # Append the anime_specific_tag at the beginning of the raw_sd_prompt
    sd_prompt = anime_specific_tag + raw_sd_prompt.strip()

    print(f"Gender: {gender}")
    print(f"SD Prompt: {sd_prompt}")

    sd_filter(nsfw_filter)
    return image_generate(character_name, sd_prompt, input_none(negative_prompt))


def image_generate(character_name, prompt, negative_prompt):
    global sd  # Make sure sd is defined globally
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

    # Generate the image
    generated_image = \
    sd(prompt, negative_prompt=negative_prompt, height=768, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Prepare the filename and directory
    character_name = character_name.replace(" ", "_")
    os.makedirs(f"characters/{character_name}", exist_ok=True)
    image_path = f"characters/{character_name}/{character_name}.png"

    # Save the generated image
    generated_image.save(image_path, format='PNG')

    # Load the image back into a NumPy array
    reloaded_image = Image.open(image_path)
    reloaded_image_np = np.array(reloaded_image)

    # Call process_uploaded_image
    process_uploaded_image(reloaded_image_np)

    print(f"Generated character avatar: {prompt}")
    return generated_image

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

def sd_filter(enable):
    global sd, safety_checker_sd
    if enable:
        sd.safety_checker = safety_checker_sd
        sd.requires_safety_checker = True
    else:
        sd.safety_checker = None
        sd.requires_safety_checker = False

# Make sure to define safety_checker_sd somewhere in your code, for example:
safety_checker_sd = sd.safety_checker