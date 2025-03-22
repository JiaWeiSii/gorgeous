import glob
import os
import random

import gradio as gr
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from torchvision.transforms import transforms
from transformers import AutoTokenizer, PretrainedConfig

from face_parsing import inference as face_parsing_inference

# ----------------------------------------------------------------

# Define model paths and other parameters

# sd 1.5
# pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# controlnet_path = "siijiawei/gorgeous-mafor-sd1-5"

# sd 2.1
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "siijiawei/gorgeous-mafor-sd2-1"

image_sets = sorted(glob.glob("makeup_assets/*"))
textual_inversion_paths = sorted(glob.glob("makeup_assets/*"))

prompt_template = "A woman with {} makeup on face"
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


# ----------------------------------------------------------------


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# ----------------------------------------------------------------

# Initialize components
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
)
text_encoder_cls = import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path, "main"
)
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    use_safetensors=True,
    torch_dtype=torch.float16,
    # subfolder="controlnet",
).to(device)

vae.to(device, dtype=dtype)
unet.to(device, dtype=dtype)
text_encoder.to(device, dtype=dtype)
controlnet.to(device, dtype=dtype)

pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=dtype,
    use_safetensors=True,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.to(device)

textual_inversion_tokens = [f"<v{i}>" for i in range(len(textual_inversion_paths))]
pipeline.load_textual_inversion(textual_inversion_paths, token=textual_inversion_tokens)

generator = torch.Generator(device=device).manual_seed(42)

preprocess_transform = transforms.Compose(
    [transforms.Resize(512), transforms.CenterCrop(512)]
)


# ----------------------------------------------------------------


# Helper functions
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# ----------------------------------------------------------------


def create_image(
    idea_set_target,
    input_image,
    prompt,
    n_prompt,
    control_scale,
    guidance_scale,
    num_inference_steps,
    seed,
):
    if input_image is not None:
        # Generate mask
        input_image_path = "input_image.png"
        input_image.save(input_image_path)

        input_image = preprocess_transform(input_image)
        mask_image = face_parsing_inference.get_face_mask(input_image).convert("L")

        print("idea_set_target", idea_set_target)

        set_index = int(idea_set_target.split(":")[0].replace("Set ", "")) - 1  # start from 1

        # Prepare prompt
        token = textual_inversion_tokens[set_index]
        prompt = prompt.replace("{}", token)
        print(prompt)

        # Generate image
        blurred_mask = pipeline.mask_processor.blur(mask_image, blur_factor=10)
        masked_image = make_inpaint_condition(input_image, blurred_mask)

        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.autocast("cuda"):
            output = pipeline(
                prompt=prompt,
                image=input_image,
                mask_image=blurred_mask,
                control_image=input_image,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
                negative_prompt=n_prompt,
                controlnet_conditioning_scale=float(control_scale),
                guidance_scale=float(guidance_scale),
            )

        output_image = output.images[0]
        return output_image
    return None


# ----------------------------------------------------------------


def read_image_from_dirpath(dirpath):
    img_paths = sorted(
        glob.glob(dirpath + "/*.png")
        + glob.glob(dirpath + "/*.jpeg")
        + glob.glob(dirpath + "/*.jpg")
    )
    imgs = [Image.open(p) for p in img_paths[:5]]

    if len(imgs) < 5:
        imgs += [Image.new(mode="RGB", size=(200, 200)) for _ in range(5 - len(imgs))]

    return imgs




image_sets = [
    {
        "label": f"Set {i + 1}: {os.path.basename(image_sets[i])}",
        "images": read_image_from_dirpath(image_sets[i]),
    }
    for i in range(len(image_sets))
]

labels = [image_set["label"] for image_set in image_sets]

def display_images(set_label):
    set_index = int(set_label.split(":")[0].replace("Set ", "")) - 1  # start from 1
    image_set = image_sets[set_index]
    return [image_set["label"]] + image_set["images"]


# ----------------------------------------------------------------

# Gradio UI setup
block = gr.Blocks(
    css="""
        footer {visibility: hidden}
        .title-background {
            background-color: #f7e4da; /* Light brown background */
            color: #1d1d1d; /* Dark text color */
            padding: 20px; /* Padding for top and bottom */
            text-align: center;
            width: 100%; /* Set width to 100% */
            margin: 0 auto; /* Center alignment */
            max-width: 1200px; /* Max width to keep content centered */
            box-sizing: border-box; /* Ensure padding is inside the box model */
        }
        .gr-button {
            background-color: #c2410c !important; /* Brown color for buttons */
            color: white !important; /* Text color */
        }
        .gr-dropdown, .gr-slider, .gr-textbox {
            border-color: #c2410c !important; /* Brown color for borders */
        }
        .gr-label, .gr-markdown {
            color: #c2410c !important; /* Brown color for text */
        }
        .content-description {
            text-align: center;
            max-width: 1200px; /* Ensure same max width as title */
            margin: 0 auto; /* Center alignment */
            box-sizing: border-box;
        }
    """
).queue(max_size=10, api_open=False)

with block:
    # Title with background
    gr.Markdown(
        """
        <div class="title-background">
            <h1 style='font-weight: 10px; font-size: 40px;'>GORGEOUS</h1>
        </div>
        """
    )
    # Description with center alignment
    gr.Markdown(
        """
        <div class="content-description">Gorgeous is a novel diffusion-based makeup application method that goes beyond simple transfer by innovatively crafting unique and thematic facial makeup. It draws artistic inspiration from a minimal set of three to five images and transforms these elements into practical makeup applications directly on the face. It can effectively generate distinctive character facial makeup inspired by the chosen thematic reference images. <br>This approach opens up new possibilities for integrating broader story elements into character makeup, thereby enhancing the narrative depth and visual impact in storytelling.</div>
        """
    )

    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_pil = gr.Image(
                        label="Targeted face (e.g., your face)", type="pil"
                    ).style(height=256)
                    generated_image = gr.Image(
                        label="Generated Image", type="pil"
                    ).style(height=256)

                with gr.Row():
                    set_dropdown = gr.Dropdown(
                        choices=[
                            labels[i]
                            for i in range(len(image_sets))
                        ],
                        label="Select Image Set",
                        value=labels[0],
                    )
                    image_label = gr.Label()
                    image_boxes = [gr.Image() for _ in range(5)]

                    set_dropdown.change(
                        display_images,
                        set_dropdown,
                        outputs=[image_label] + image_boxes,
                    )

                with gr.Row():
                    scale = gr.Slider(
                        minimum=0,
                        maximum=30,
                        step=0.01,
                        value=20.0,
                        label="Guidance scale (Adjust the slider to steer the influence of the idea chosen on the generation.)",
                    )
                    control_scale = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=1,
                        label="Control scale (Adjust the slider to control face fidelity.)",
                    )
                    num_inference_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=50,
                        label="Number of inference steps",
                    )

                # prompt_template = "A woman with {} makeup on face"

                with gr.Row():
                    prompt = gr.Textbox(
                        label='Prompt (the set is represented by "{}")',
                        value="A photo of a woman with {} on face",
                    )

                with gr.Row():
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting",
                    )

                with gr.Row():
                    seed = gr.Slider(
                        minimum=0, maximum=MAX_SEED, value=1, step=1, label="Seed Value"
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                generate_button = gr.Button("Generate Image")

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[
                set_dropdown,
                image_pil,
                prompt,
                n_prompt,
                control_scale,
                scale,
                num_inference_steps,
                seed,
            ],
            outputs=generated_image,
        )

    gr.Markdown("### Article")

block.launch(share=True)
