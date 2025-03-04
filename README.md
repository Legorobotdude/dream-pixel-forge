# DreamPixelForge

A modern GUI application for running Stable Diffusion locally on your machine, transforming your text prompts into stunning images.

## Features

- Clean and intuitive user interface
- Text-to-image generation using Stable Diffusion
- Support for negative prompts
- Adjustable generation parameters (steps and guidance scale)
- Save generated images in various formats
- GPU acceleration support (if available)
- Customizable resolution presets
- Real-time progress tracking

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but not required)
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
2. Enter your prompt in the text field
3. (Optional) Enter a negative prompt to specify what you don't want in the image
4. Adjust the generation parameters if needed:
   - Number of Steps: Higher values (30-50) generally produce better quality but take longer
   - Guidance Scale: Higher values (7.5-15) make the image more closely match the prompt
5. Click "Generate Image" and wait for the result
6. Use the "Save Image" button to save your generated image

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

- The first run will download the Stable Diffusion model (about 4GB)
- Generation time depends on your hardware (GPU recommended)
- If you don't have a GPU, the application will run on CPU but will be significantly slower

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed correctly
2. Check if you have enough disk space for the model
3. If using GPU, ensure you have the latest CUDA drivers installed
4. Try reducing the number of steps or image size if you run into memory issues 