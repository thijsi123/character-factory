

safety_checker_sd = None

import requests
from diffusers import StableDiffusionPipeline
import torch
import gradio as gr
import re
from PIL import Image
import numpy as np
import gc  # Python's garbage collector

llm = None
sd = None
safety_checker_sd = None

folder_path = "models"  # Base directory for models

def load_model(model_name, use_safetensors=False, use_local=False):
    global sd

    # Enable TensorFloat-32 for matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = True

    if use_local:
        model_path = os.path.join(folder_path, model_name).replace("\\", "/")
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}.")
            return
        print(f"Loading local model from: {model_path}")
        sd = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)
    else:
        print(f"Loading {model_name} from Hugging Face with safetensors={use_safetensors}.")
        sd = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=use_safetensors, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        sd.to("cuda")
        print(f"Loaded {model_name} to GPU in half precision (float16).")
    else:
        print(f"Loaded {model_name} to CPU.")


def unload_model():
    global sd
    if sd is not None:
        del sd  # Delete the model object
        sd = None  # Set the global variable to None
    gc.collect()  # Call the garbage collector

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
Danbooru tags are descriptive labels used on the Danbooru image board to categorize and describe images, especially in anime and manga art. They cover a wide range of specifics such as character features, clothing styles, settings, and themes. These tags help in organizing and navigating the large volume of artwork and are also useful in guiding AI models like Stable Diffusion to generate specific images.
A brief example of Danbooru tags for an anime character might look like this:

Appearance: blue_eyes, short_hair, smiling
Clothing: school_uniform, necktie
Setting: classroom, daylight
In this example, the tags precisely describe the character's appearance (blue eyes, short hair, smiling), their clothing (school uniform with a necktie), and the setting of the image (classroom during daylight). 
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
    # Detect if "anime" is in the character summary or topic and adjust the prompt
    anime_specific_tag = "anime, 2d, " if 'anime' in character_summary.lower() or 'anime' in topic.lower() else ""
    raw_sd_prompt = (
            input_none(avatar_prompt)
            or send_message(
        example_dialogue
        + "\n<|user|> create a prompt that lists the appearance "
        + "characteristics of a character whose summary is "
        + "if lack of info, generate something based on available info."
        + f"{character_summary}. Topic: {topic}</s>\n<|assistant|> "
    ).strip()
    )
    # Append the anime_specific_tag at the beginning of the raw_sd_prompt
    sd_prompt = anime_specific_tag + raw_sd_prompt.strip()
    print(sd_prompt)
    sd_filter(nsfw_filter)
    return image_generate(character_name,
                          sd_prompt,
                          input_none(negative_prompt),
                          )

def image_generate(character_name, prompt, negative_prompt):
    print("Loading model")
    # For a local .safetensors model
    load_model("oof.safetensors", use_safetensors=True, use_local=True)
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

    image_path = f"characters/{character_name}/{character_name}.png"

    # Save the generated image
    generated_image.save(image_path)

    # Load the image back into a NumPy array
    reloaded_image = Image.open(image_path)
    reloaded_image_np = np.array(reloaded_image)

    # Call process_uploaded_image
    process_uploaded_image(reloaded_image_np)

    print("Generated character avatar" + prompt)

    print("Unloading model")
    unload_model()

    return generated_image

def sd_filter(enable):
    if enable:
        sd.safety_checker = safety_checker_sd
        sd.requires_safety_checker = True
    else:
        sd.safety_checker = None
        sd.requires_safety_checker = False

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

safety_checker_sd = sd.safety_checker