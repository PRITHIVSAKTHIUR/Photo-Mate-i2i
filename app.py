import os
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from aura_sr import AuraSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- # Device and CUDA Setup Check ---
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --- Main Model Initialization ---
MAX_SEED = np.iinfo(np.int32).max
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")

# --- Load All Adapters ---
pipe.load_lora_weights("prithivMLmods/PhotoCleanser-i2i", weight_name="PhotoCleanser-i2i.safetensors", adapter_name="cleanser")
pipe.load_lora_weights("prithivMLmods/Photo-Restore-i2i", weight_name="Photo-Restore-i2i.safetensors", adapter_name="restorer")
pipe.load_lora_weights("prithivMLmods/Polaroid-Warm-i2i", weight_name="Polaroid-Warm-i2i.safetensors", adapter_name="polaroid")
pipe.load_lora_weights("prithivMLmods/Monochrome-Pencil", weight_name="Monochrome-Pencil-i2i.safetensors", adapter_name="pencil")
pipe.load_lora_weights("prithivMLmods/LZO-1-Preview", weight_name="LZO-1-Preview.safetensors", adapter_name="lzo")
pipe.load_lora_weights("prithivMLmods/Kontext-Watermark-Remover", weight_name="Kontext-Watermark-Remover.safetensors", adapter_name="watermark-remover")
pipe.load_lora_weights("prithivMLmods/Kontext-Unblur-Upscale", weight_name="Kontext-Image-Upscale.safetensors", adapter_name="unblur-upscale")

# --- Upscaler Model Initialization ---
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

@spaces.GPU
def infer(input_image, prompt, lora_adapter, upscale_image, seed=42, randomize_seed=False, guidance_scale=2.5, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing and optional upscaling, returning the final image.
    """
    if not input_image:
        raise gr.Error("Please upload an image for editing.")

    if lora_adapter == "PhotoCleanser":
        pipe.set_adapters(["cleanser"], adapter_weights=[1.0])
    elif lora_adapter == "PhotoRestorer":
        pipe.set_adapters(["restorer"], adapter_weights=[1.0])
    elif lora_adapter == "PolaroidWarm":
        pipe.set_adapters(["polaroid"], adapter_weights=[1.0])
    elif lora_adapter == "MonochromePencil":
        pipe.set_adapters(["pencil"], adapter_weights=[1.0])
    elif lora_adapter == "LZO-Zoom":
        pipe.set_adapters(["lzo"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-Watermark-Remover":
        pipe.set_adapters(["watermark-remover"], adapter_weights=[1.0])
    elif lora_adapter == "Kontext-Unblur-Upscale":
        pipe.set_adapters(["unblur-upscale"], adapter_weights=[1.0])
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    original_image = input_image.copy().convert("RGB")
    
    image = pipe(
        image=original_image, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        width = original_image.size[0],
        height = original_image.size[1],
        num_inference_steps=steps,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    if upscale_image:
        progress(0.8, desc="Upscaling image...")
        image = aura_sr.upscale_4x(image)

    return image, seed, gr.Button(visible=True)

@spaces.GPU
def infer_example(input_image, prompt, lora_adapter):
    """
    Wrapper function for gr.Examples.
    """
    image, seed, _ = infer(input_image, prompt, lora_adapter, upscale_image=False)
    return image, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.2em !important;}
"""

with gr.Blocks() as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# **Photo-Mate-i2i**", elem_id="main-title")
        gr.Markdown("Image manipulation with FLUX.1 Kontext adapters. [How to Use](https://huggingface.co/spaces/prithivMLmods/Photo-Mate-i2i/discussions/2) [[Version 2.0]](https://huggingface.co/spaces/prithivMLmods/Kontext-Photo-Mate-v2)")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil", height=290)
                
                prompt = gr.Text(
                    label="Edit Prompt",
                    show_label=True,
                    placeholder="e.g., transform into anime..",
                )

                run_button = gr.Button("Edit Image", variant="primary")

                with gr.Accordion("Advanced Settings", open=False):
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )       
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=30,
                        value=28,
                        step=1
                    )
                    
            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=355)
                reuse_button = gr.Button("Reuse this image", visible=False)
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Chosen LoRA",
                        choices=["PhotoCleanser", "PhotoRestorer", "PolaroidWarm", "MonochromePencil", "LZO-Zoom", "Kontext-Watermark-Remover", "Kontext-Unblur-Upscale"],
                        value="PhotoCleanser"
                    )
                    
                with gr.Row():
                    upscale_checkbox = gr.Checkbox(label="Upscale the final image", value=False)

        gr.Examples(
            examples=[
                ["photocleanser/2.png", "[photo content], remove the cat from the image while preserving the background and remaining elements, maintaining realism and original details.", "PhotoCleanser"],
                ["photocleanser/1.png", "[photo content], remove the football from the image while preserving the background and remaining elements, maintaining realism and original details.", "PhotoCleanser"],
                ["watermark/12.jpeg", "[photo content], remove any watermark text or logos from the image while preserving the background, texture, lighting, and overall realism. Ensure the edited areas blend seamlessly with surrounding details, leaving no visible traces of watermark removal.", "Kontext-Watermark-Remover"],
                ["photorestore/1.png", "[photo content], restore and enhance the image by repairing any damage, scratches, or fading. Colorize the photo naturally while preserving authentic textures and details, maintaining a realistic and historically accurate look.", "PhotoRestorer"],
                ["lzo/1.jpg", "[photo content], zoom in on the specified [face close-up], enhancing resolution and detail while preserving sharpness, realism, and original context. Maintain natural proportions and background continuity around the zoomed area.", "LZO-Zoom"],
                ["photorestore/2.png", "[photo content], restore and enhance the image by repairing any damage, scratches, or fading. Colorize the photo naturally while preserving authentic textures and details, maintaining a realistic and historically accurate look.", "PhotoRestorer"],
                ["polaroid/1.png", "[photo content], in the style of a vintage Polaroid, with warm, faded tones, and a white border.", "PolaroidWarm"],
                ["unblur/1.jpg", "[photo content], upscale the low-quality image to 4K resolution, enhancing sharpness, clarity, and fine details while preserving the original texture, colors, lighting, and natural appearance. Remove noise, blur, and compression artifacts without over-smoothing or distorting facial or object features. Ensure realistic depth, balanced contrast, and accurate tones, achieving a high-definition, lifelike result that maintains the integrity of the original image.", "Kontext-Unblur-Upscale"],
                ["pencil/1.png", "[photo content], replicate the image as a pencil illustration, black and white, with sketch-like detailing.", "MonochromePencil"],
                ["unblur/11.jpg", "[photo content], upscale the low-quality image to 4K resolution, enhancing sharpness, clarity, and fine details while preserving the original texture, colors, lighting, and natural appearance. Remove noise, blur, and compression artifacts without over-smoothing or distorting facial or object features. Ensure realistic depth, balanced contrast, and accurate tones, achieving a high-definition, lifelike result that maintains the integrity of the original image.", "Kontext-Unblur-Upscale"],
            ],
            inputs=[input_image, prompt, lora_adapter],
            outputs=[output_image, seed],
            fn=infer_example,
            cache_examples=False,
            label="Examples"
        )
            
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[input_image, prompt, lora_adapter, upscale_checkbox, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed, reuse_button]
    )
    
    reuse_button.click(
        fn=lambda x: x,
        inputs=[output_image],
        outputs=[input_image]
    )

demo.launch(css=css, theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True)
