# **Photo-Mate-i2i**

Photo-Mate-i2i is a Hugging Face Space for experimenting with adapters for image manipulation using FLUX.1 Kontext adapters. It includes specialized LoRA adapters such as Photo-Restore-i2i, PhotoCleanser-i2i, Polaroid-Warm-i2i, Monochrome-Pencil-i2i, and more. This app allows users to upload images, apply edits via prompts, and enhance them with various styles or restorations.

## Features

- **Image Editing with Prompts**: Upload an image and use natural language prompts to edit it (e.g., "Remove glasses" or "Add a hat").
- **LoRA Adapters**: Choose from multiple adapters for different effects:
  - **PhotoCleanser**: Clean and remove unwanted elements while preserving realism.
  - **PhotoRestorer**: Restore old or damaged photos, repair scratches, and colorize naturally.
  - **PolaroidWarm**: Apply a vintage Polaroid style with warm, faded tones and borders.
  - **MonochromePencil**: Convert images to black-and-white pencil sketches.
- **Upscaling**: Optional 4x upscaling using AuraSR for higher resolution outputs.
- **Advanced Controls**: Adjust seed, guidance scale, inference steps, and randomize seed for reproducibility and variation.
- **Before/After Slider**: Interactive slider to compare original and edited images.
- **Examples**: Pre-loaded examples to demonstrate each adapter's capabilities.
- **Reuse Functionality**: Easily reuse the generated image as input for further edits.

## How to Use

1. **Upload an Image**: Select or drag an image into the input field.
2. **Enter a Prompt**: Describe the desired edit (e.g., "[photo content], remove the cat from the image while preserving the background").
3. **Choose an Adapter**: Select a LoRA adapter from the dropdown.
4. **Optional Settings**:
   - Enable upscaling for a higher-resolution result.
   - Adjust advanced parameters like seed, guidance scale, and steps under the accordion.
5. **Run**: Click "Run" or press Enter in the prompt field to generate the edited image.
6. **View Results**: Use the slider to compare before and after. Click "Reuse this image" to set the output as the new input.
7. **Examples**: Click on the provided examples to quickly test different adapters and prompts.

For more details on usage, check the discussion: [How to Use](https://huggingface.co/spaces/prithivMLmods/Photo-Mate-i2i/discussions/2).


## Sample Inferences

| ![Screenshot 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/eDELepJP67sngtcN1wejA.png) | ![Screenshot 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/caumfbjUrc0BXrX7FYReT.png) |
|---|---|
| ![Screenshot 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/5yST1XqMRgOSBiWklCF3i.png) | ![Screenshot 4](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/wEU4_fQEVWCAmxN5Lurwd.png) |

| ![Screenshot 5](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/IKKB83XBO_fTTuzhu14cl.png) |
|---|

## Requirements

This app runs on Hugging Face Spaces with GPU acceleration. It uses:
- Diffusers library with FLUX.1-Kontext-dev model.
- LoRA weights from [prithivMLmods](https://huggingface.co/prithivMLmods) collections.
- AuraSR for upscaling.
- Gradio for the interactive UI.

No local installation is required to use the hosted Space. If running locally:

### Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Photo-Mate-i2i.git
   cd Photo-Mate-i2i
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Assuming a `requirements.txt` with packages like `gradio`, `diffusers`, `torch`, `spaces`, `huggingface_hub`, `aura_sr`, etc.)

3. Run the app:
   ```
   python app.py
   ```

Note: Requires a CUDA-enabled GPU for optimal performance. The app checks for CUDA availability and falls back to CPU if needed.

## Credits

- Built with [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) by Black Forest Labs.
- LoRA adapters by [prithivMLmods](https://huggingface.co/prithivMLmods). [Myself]
- Upscaling powered by [AuraSR-v2](https://huggingface.co/fal/AuraSR-v2).
- UI powered by [Gradio](https://gradio.app/) and [Hugging Face Spaces](https://huggingface.co/spaces).
- GitHub: [https://github.com/PRITHIVSAKTHIUR/Photo-Mate-i2i](https://github.com/PRITHIVSAKTHIUR/Photo-Mate-i2i)


For issues or contributions, visit the GitHub repository.
