import os
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline
import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
        sd = StableDiffusionPipeline.from_pretrained(model_name, use_safetensors=use_safetensors,
                                                     torch_dtype=torch.float16)

    if torch.cuda.is_available():
        sd.to("cuda")
        print(f"Loaded {model_name} to GPU in half precision (float16).")
    else:
        print(f"Loaded {model_name} to CPU.")

# Load the "oof.safetensors" model from a local directory
load_model("oof.safetensors", use_safetensors=True, use_local=True)

# Initialize the StableDiffusionPipeline
pipeline = sd

# Load the local image
input_image_path = "00006-3480988086.png"
input_image = Image.open(input_image_path).convert("RGB")

# Define the prompt and the generator
prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
generator = torch.manual_seed(33)

# Generate the low-resolution latents
low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

# Define the prompts
prompts = [
    "an old lady",
]

# Loop over the prompts
for prompt in prompts:
    # Upscale the image using the defined prompt
    upscaled_image = pipeline(prompt=prompt).images[0]
    upscaled_image.show()  # Display the upscaled image
