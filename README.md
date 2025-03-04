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
- Prompt enhancement using local LLMs via Ollama
- GPU acceleration support (if available)
- Real-time progress tracking
- Clear feedback during model downloads

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8+ GB VRAM for SDXL)
- At least 8GB of RAM (16GB recommended)
- 4-7GB free disk space per model (~20GB for all models)

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

## Ollama Prompt Enhancement

DreamPixelForge supports prompt enhancement using local large language models via the Ollama project. This feature helps you:

1. Convert descriptive sentences into a concise set of 5-10 optimized image generation tags
2. Enhance existing tags with 3-5 additional very closely related keywords that improve your results while maintaining the original style and concept

### Requirements for Ollama Integration

- [Ollama](https://ollama.ai/) installed and running on your machine
- At least one language model installed through Ollama

### Setting up Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai/)
2. Run Ollama according to the instructions for your operating system
3. Pull a language model using the Ollama command:
   ```bash
   ollama pull llama2
   ```

### Using Prompt Enhancement

1. Choose an Ollama model from the dropdown
2. Select the input type:
   - **Description to Tags**: Enter a full description of what you want to see
   - **Enhance Tags**: Enter existing tags/keywords to expand them
3. Type your prompt in the enhancement field
4. Click "Enhance Prompt" to process it through the selected LLM
5. The enhanced prompt will be placed in the main prompt field, ready for image generation

**Note:** You need to start Ollama separately before using this feature. If Ollama is not detected, a message will be shown with an option to check for availability.

## Model Downloads and First Use

When you first use a model, it will be downloaded automatically from Hugging Face. The application will show:
1. A first-time use notice with download size information
2. Real-time download status in the progress area
3. Elapsed time for longer downloads

Download sizes for each model:
- Stable Diffusion 1.5: ~4GB
- Stable Diffusion 2.1: ~4.2GB
- Dreamlike Diffusion: ~4GB
- Kandinsky 2.2: ~4.5GB
- Stable Diffusion XL: ~6.5GB

**Note:** Downloads happen only once per model. After downloading, the model will be loaded directly from your local cache.

## Model Storage and Cache Management

Models are downloaded automatically by the Hugging Face Diffusers library when first used and stored in a cache directory:

- **Windows**: `C:\Users\<YOUR_USERNAME>\.cache\huggingface\hub`
- **macOS**: `/Users/<YOUR_USERNAME>/.cache/huggingface/hub`
- **Linux**: `/home/<YOUR_USERNAME>/.cache/huggingface/hub`

### Disk Space Requirements

Each model requires significant disk space:
- Stable Diffusion 1.5/2.1: ~4GB each
- Dreamlike Diffusion: ~4GB
- Kandinsky 2.2: ~4-5GB
- Stable Diffusion XL: ~6.5GB

**Note:** You only need disk space for the models you actually use. The ~20GB total is only if you plan to use all models. Most users will only need 4-7GB for their preferred model.

### Managing the Cache

You can manage the model cache in several ways:

1. **Clear the cache** - You can safely delete the cache directory if you need to free up space. Models will be re-downloaded when needed.

2. **Custom cache location** - Set a custom cache directory by setting the `HF_HOME` environment variable before running the application:

   ```bash
   # Windows (PowerShell)
   $env:HF_HOME = "D:\custom_model_cache"
   python dream_pixel_forge.py

   # Linux/macOS
   export HF_HOME="/path/to/custom_model_cache"
   python dream_pixel_forge.py
   ```

3. **One-time downloads** - Models are only downloaded once, so subsequent runs will be faster.

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