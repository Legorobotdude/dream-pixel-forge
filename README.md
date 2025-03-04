# DreamPixelForge

A modern GUI application for running multiple AI image generation models locally on your machine, transforming your text prompts into stunning images.

## Features

- Clean and intuitive user interface
- Multi-model support:
  - Stable Diffusion 1.5
  - Stable Diffusion 2.1
  - Stable Diffusion XL
  - Dreamlike Diffusion
  - Kandinsky 2.2
- Text-to-image generation with various models
- Support for negative prompts
- Adjustable generation parameters (steps and guidance scale)
- Model-specific resolution presets
- Save generated images in various formats
- GPU acceleration support (if available)
- Real-time progress tracking

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8+ GB VRAM for SDXL)
- At least 8GB of RAM (16GB recommended)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dream-pixel-forge.git
   cd dream-pixel-forge
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python dream_pixel_forge.py
   ```
2. Select the model you want to use from the dropdown menu
3. Enter your prompt in the text field
4. (Optional) Enter a negative prompt to specify what you don't want in the image
5. Adjust the generation parameters if needed:
   - Number of Steps: Higher values (30-50) generally produce better quality but take longer
   - Guidance Scale: Higher values (7.5-15) make the image more closely match the prompt
6. Click "Generate Image" and wait for the result
7. Use the "Save Image" button to save your generated image

## Models

### Stable Diffusion 1.5
The original Stable Diffusion model - fast and versatile.

### Stable Diffusion 2.1
Improved version with better quality and consistency.

### Stable Diffusion XL
Larger model with higher quality outputs (requires more VRAM).

### Dreamlike Diffusion
Artistic model that creates dreamlike, surreal images.

### Kandinsky 2.2
Russian alternative to SD with unique artistic style.

## Development

This project uses Git for version control. After making changes:

```bash
# View changed files
git status

# Add files to staging area
git add .

# Commit changes with a descriptive message
git commit -m "Description of changes"

# Push changes to remote repository (if set up)
git push
```

## Notes

- The first run will download the selected model (SD models ~4GB, SDXL ~6.5GB)
- Generation time depends on your hardware (GPU recommended)
- If you don't have a GPU, the application will run on CPU but will be significantly slower
- Different models have different VRAM requirements:
  - SD 1.5 and 2.1: ~4GB VRAM
  - Dreamlike/specialized models: ~4-6GB VRAM
  - SDXL: 8+GB VRAM recommended

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed correctly
2. Check if you have enough disk space and VRAM for the selected model
3. If using GPU, ensure you have the latest CUDA drivers installed
4. Try reducing the number of steps or image size if you run into memory issues
5. For SDXL, you need a GPU with at least 8GB VRAM, or consider using CPU mode 