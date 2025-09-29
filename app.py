import os
import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image

from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from aura_sr import AuraSR
from gradio_imageslider import ImageSlider

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

# --- Main Model Initialization ---
MAX_SEED = np.iinfo(np.int32).max
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")

# --- Load All Adapters ---
pipe.load_lora_weights("prithivMLmods/PhotoCleanser-i2i", weight_name="PhotoCleanser-i2i.safetensors", adapter_name="cleanser")
pipe.load_lora_weights("prithivMLmods/Photo-Restore-i2i", weight_name="Photo-Restore-i2i.safetensors", adapter_name="restorer")
pipe.load_lora_weights("prithivMLmods/Polaroid-Warm-i2i", weight_name="Polaroid-Warm-i2i.safetensors", adapter_name="polaroid")
pipe.load_lora_weights("prithivMLmods/Monochrome-Pencil", weight_name="Monochrome-Pencil-i2i.safetensors", adapter_name="pencil")
# Add the new LZO adapter
pipe.load_lora_weights("prithivMLmods/LZO-1-Preview", weight_name="LZO-1-Preview.safetensors", adapter_name="lzo")


# --- Upscaler Model Initialization ---
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

@spaces.GPU
def infer(input_image, prompt, lora_adapter, upscale_image, seed=42, randomize_seed=False, guidance_scale=2.5, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing and optional upscaling, returning a pair for the ImageSlider.
    
    Returns:
        tuple: A 3-tuple containing:
               - (PIL.Image.Image, PIL.Image.Image): A tuple of the (original, generated) images for the slider.
               - int: The seed used for generation.
               - gr.update: A Gradio update to make the reuse button visible.
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
    # Add the new LZO adapter condition
    elif lora_adapter == "LZO-Zoom":
        pipe.set_adapters(["lzo"], adapter_weights=[1.0])


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

    return (original_image, image), seed, gr.Button(visible=True)

@spaces.GPU
def infer_example(input_image, prompt, lora_adapter):
    """
    Wrapper function for gr.Examples to call the main infer logic for the slider.
    """
    (original_image, generated_image), seed, _ = infer(input_image, prompt, lora_adapter, upscale_image=False)
    return (original_image, generated_image), seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
"""

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# **[Photo-Mate-i2i](https://huggingface.co/collections/prithivMLmods/i2i-kontext-exp-68ce573b5c0623476b636ec7)**
        Image manipulation with FLUX.1 Kontext adapters. [How to Use](https://huggingface.co/spaces/prithivMLmods/Photo-Mate-i2i/discussions/2)""")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Image [PIL]", type="pil", height="300")
                with gr.Row():
                    prompt = gr.Text(
                        label="Edit Prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt for editing (e.g., 'Remove glasses', 'Add a hat')",
                        container=False,
                    )
                    run_button = gr.Button("Run", elem_classes="submit-btn", scale=0)
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
                # Replace the single image result with the ImageSlider
                output_slider = ImageSlider(label="Before / After", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image", visible=False)
                
                with gr.Row():
                    lora_adapter = gr.Dropdown(
                    label="Chosen LoRA",
                    choices=["PhotoCleanser", "PhotoRestorer", "PolaroidWarm", "MonochromePencil", "LZO-Zoom"],
                    value="PhotoCleanser"
                )
                    
                #AuraSR Upscale
                with gr.Row():
                    upscale_checkbox = gr.Checkbox(label="Upscale the final image", value=False)

        gr.Examples(
            examples=[
                ["photocleanser/2.png", "[photo content], remove the cat from the image while preserving the background and remaining elements, maintaining realism and original details.", "PhotoCleanser"],
                ["photocleanser/1.png", "[photo content], remove the football from the image while preserving the background and remaining elements, maintaining realism and original details.", "PhotoCleanser"],
                ["photorestore/1.png", "[photo content], restore and enhance the image by repairing any damage, scratches, or fading. Colorize the photo naturally while preserving authentic textures and details, maintaining a realistic and historically accurate look.", "PhotoRestorer"],
                # Add the new LZO example
                ["lzo/1.jpg", "[photo content], zoom in on the specified [face close-up], enhancing resolution and detail while preserving sharpness, realism, and original context. Maintain natural proportions and background continuity around the zoomed area.", "LZO-Zoom"],
                ["photorestore/2.png", "[photo content], restore and enhance the image by repairing any damage, scratches, or fading. Colorize the photo naturally while preserving authentic textures and details, maintaining a realistic and historically accurate look.", "PhotoRestorer"],
                ["polaroid/1.png", "[photo content], in the style of a vintage Polaroid, with warm, faded tones, and a white border.", "PolaroidWarm"],
                ["pencil/1.png", "[photo content], replicate the image as a pencil illustration, black and white, with sketch-like detailing.", "MonochromePencil"],

            ],
            inputs=[input_image, prompt, lora_adapter],
            # The output now targets the ImageSlider component
            outputs=[output_slider, seed],
            fn=infer_example,
            cache_examples="lazy",
            label="Examples"
        )
            
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[input_image, prompt, lora_adapter, upscale_checkbox, seed, randomize_seed, guidance_scale, steps],
        # The output now targets the ImageSlider component
        outputs=[output_slider, seed, reuse_button]
    )
    
    # Update the reuse function to handle the ImageSlider output
    reuse_button.click(
        # The slider outputs a tuple of images; we want the second one (the generated result)
        fn=lambda images: images[1] if isinstance(images, (list, tuple)) and len(images) > 1 else images,
        inputs=[output_slider],
        outputs=[input_image]
    )

demo.launch(mcp_server=True, ssr_mode=False, show_error=True)
