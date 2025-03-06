import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
                            QComboBox, QMessageBox, QGroupBox, QRadioButton, 
                            QTabWidget, QListWidget, QListWidgetItem, QDialog,
                            QFormLayout, QFrame, QStyle, QStyleOptionComboBox,
                            QMenu, QCheckBox, QSizePolicy, QGridLayout,
                            QScrollArea, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QAction, QPainter, QColor, QFont, QFontMetrics, QCursor
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, KandinskyV22Pipeline, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, PNDMScheduler, DDPMScheduler, DEISMultistepScheduler, DPMSolverSDEScheduler, KarrasVeScheduler
import torch
from PIL import Image, ImageDraw, ImageChops
import io
import traceback
from huggingface_hub import scan_cache_dir, HfFolder, model_info
import time
import requests
import json
import random
import glob
import platform
import logging

# Local models directory
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# Detect macOS for MPS (Metal) support
IS_MACOS = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
HAS_MPS = False
if IS_MACOS:
    try:
        HAS_MPS = torch.backends.mps.is_available()
        if HAS_MPS:
            print("MPS (Metal Performance Shaders) is available on this Mac")
        else:
            print("MPS is not available on this Mac")
    except:
        print("Could not check MPS availability, assuming it's not available")
        HAS_MPS = False

# Function to get the appropriate device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif HAS_MPS:
        return "mps"
    else:
        return "cpu"

# Function to get appropriate torch dtype based on device
def get_torch_dtype():
    device = get_device()
    if device == "cuda":
        return torch.float16
    elif device == "mps":
        # Some models may work with float16 on MPS, but float32 is safer
        return torch.float32
    else:
        return torch.float32

class OllamaClient:
    """
    Client for interacting with the Ollama API to get available models and process prompts.
    """
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        
    def list_models(self):
        """Get list of models available in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                return []
        except Exception as e:
            print(f"Error getting Ollama models: {str(e)}")
            return []
            
    def is_available(self):
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
            
    def enhance_prompt(self, model, prompt, mode="tags"):
        """
        Enhance a prompt using an Ollama model.
        
        Args:
            model: The Ollama model to use
            prompt: The user's prompt
            mode: "tags" or "description" - determines the system prompt
            
        Returns:
            Enhanced prompt as a string
        """
        try:
            if mode == "tags":
                system_prompt = "You are a helpful assistant specialized in enhancing image generation prompts. The user will provide tags or keywords for an image. Respond ONLY with the original tags plus 3-5 additional very closely related tags that would improve the image generation. Focus on maintaining the exact same style and concept, just adding a few highly relevant terms. Format all tags as a single comma-separated list. Do not include explanations or other text in your response. Keep the style consistent with the original prompt."
            else:  # description
                system_prompt = "You are a helpful assistant specialized in converting descriptive text into image generation tags. The user will provide a description of an image. Respond only with a concise list of 5-10 essential tags/keywords that would help generate this image, separated by commas. Focus only on the most important visual elements in the description. Optimize the tags for image generation quality. Format your response as a simple comma-separated list with no other text or explanations."
            
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/chat", json=data)
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

class OllamaThread(QThread):
    """Thread for Ollama operations to keep UI responsive"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, client, model, prompt, mode):
        super().__init__()
        self.client = client
        self.model = model
        self.prompt = prompt
        self.mode = mode
        
    def run(self):
        try:
            result = self.client.enhance_prompt(self.model, self.prompt, self.mode)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# Local model registry to store information about local models
LOCAL_MODELS = {}  # Will be populated when local models are added

# Model-specific default negative prompts
DEFAULT_NEGATIVE_PROMPTS = {
    "sd15": "ugly, blurry, poor quality, distorted, deformed, disfigured, poorly drawn face, poorly drawn hands, poorly drawn feet, poorly drawn legs, deformed, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad proportions, gross proportions, duplicate, morbid, mutilated, extra fingers, fused fingers, too many fingers, long neck, bad composition, bad perspective, bad lighting, watermark, signature, text, logo, banner, extra digits, mutated hands and fingers",
    
    "sd21": "ugly, blurry, poor quality, distorted, deformed, disfigured, poorly drawn face, poorly drawn hands, poorly drawn feet, poorly drawn legs, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, disgusting, bad proportions, duplicate, morbid, mutilated, extra fingers, fused fingers, too many fingers, long neck, bad composition, watermark, signature, text, logo, banner, web address, artist name",
    
    "sdxl": "ugly, blurry, deformed, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, disconnected limbs, bad proportions, malformed limbs, missing arms, extra arms, missing legs, extra legs, mutated hands, fused fingers, too many fingers, missing fingers, extra fingers, poorly drawn hands, poorly drawn face, poorly drawn feet, mutation, duplicate, morbid, mutilated, poorly drawn body, dehydrated, out of frame, bad art, bad photography, pixelated, jpeg artifacts, signature, watermark, username, artist name, text, error, out of focus, lowres",
    
    "dreamlike": "ugly, blurry, watermark, text, error, pixelated, artifacting, grains, low-resolution, distorted, deformed, distorted proportions, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, low quality, gross, distasteful, bad lighting, low contrast, underexposed, overexposed, semi-realistic, child",
    
    "kandinsky": "low quality, bad quality, sketches, bad anatomy, poor detail, missing detail, missing limbs, bad proportions, deformed face, deformed hands, deformed limbs, extra limbs, too many limbs, extra fingers, too many fingers, fused fingers, watermark, signature, disfigured, duplicate, cropped, blurry, bad art, poorly drawn face, ugly, deformed eyes, text, letters",
    
    "pony": "bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, mutated hands, mutated legs, more than 2 thighs, multiple thighs, cropped, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutilated, poorly drawn hands, poorly drawn face, poorly drawn feet, poorly drawn legs, blurry, blurred, text, signature, watermark, artist name, logo, duplicate, disfigured, deformed, extra limbs, low resolution, jpeg artifacts, score_4, score_5, score_6"
}

# Use consistent variable name for backward compatibility
DEFAULT_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPTS["sd15"]

# Resolution presets that work well with SD v1.5
RESOLUTION_PRESETS = {
    "512x512 (Square)": (512, 512),
    "512x768 (Portrait)": (512, 768),
    "768x512 (Landscape)": (768, 512),
    "576x832 (Portrait)": (576, 832),
    "832x576 (Landscape)": (832, 576),
    "640x640 (Square)": (640, 640),
    "640x960 (Portrait)": (640, 960),
    "960x640 (Landscape)": (960, 640),
}

# SDXL specific resolutions
SDXL_RESOLUTION_PRESETS = {
    "1024x1024 (Square)": (1024, 1024),
    "896x1152 (Portrait)": (896, 1152),
    "1152x896 (Landscape)": (1152, 896),
    "832x1216 (Portrait)": (832, 1216),
    "1216x832 (Landscape)": (1216, 832),
}

# Model base types for local models
MODEL_TYPES = {
    "Stable Diffusion 1.5": {
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 7.5,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sd15"]
    },
    "Stable Diffusion 2.1": {
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 7.5,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sd21"]
    },
    "Stable Diffusion XL": {
        "pipeline": StableDiffusionXLPipeline,
        "resolution_presets": SDXL_RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 9.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sdxl"]
    }
}

class LocalModelInfo:
    """Class to store information about a local model"""
    def __init__(self, name, file_path, model_type="Stable Diffusion 1.5", description=""):
        self.name = name
        self.file_path = file_path
        self.model_type = model_type
        self.description = description or f"Local model: {name} ({model_type})"
        
    def get_config(self):
        """Get model configuration compatible with AVAILABLE_MODELS"""
        base_config = MODEL_TYPES[self.model_type].copy()
        
        # Keep all fields from the base config, including negative_prompt
        config = {
            "model_id": self.file_path,  # Use file path as the ID
            "description": self.description,
            "is_local": True,  # Flag to identify this is a local model
            **base_config
        }
        
        # Apply Pony model override if this is flagged as a Pony model
        if hasattr(self, 'pony_override') and self.pony_override:
            config["negative_prompt"] = DEFAULT_NEGATIVE_PROMPTS["pony"]
        
        return config

# Available samplers
SAMPLERS = {
    "Euler a": "euler_ancestral",
    "Euler": "euler",
    "LMS": "lms",
    "Heun": "heun",
    "DPM2": "dpm_2",
    "DPM2 a": "dpm_2_ancestral",
    "DPM++ 2S a": "dpmpp_2s_ancestral",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ SDE": "dpmpp_sde",
    "DDIM": "ddim"
}

# Define the available models
AVAILABLE_MODELS = {
    "Stable Diffusion 1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 7.5,
        "description": "Original Stable Diffusion model - fast and versatile",
        "size_gb": 4.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sd15"]
    },
    "Stable Diffusion 2.1": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 7.5,
        "description": "Improved version with better quality and consistency",
        "size_gb": 4.2,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sd21"]
    },
    "Stable Diffusion XL": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": StableDiffusionXLPipeline,
        "resolution_presets": SDXL_RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 9.0,
        "description": "Larger model with higher quality outputs (needs more VRAM)",
        "size_gb": 6.5,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["sdxl"]
    },
    "Dreamlike Diffusion": {
        "model_id": "dreamlike-art/dreamlike-diffusion-1.0",
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 8.0,
        "description": "Artistic model that creates dreamlike, surreal images",
        "size_gb": 4.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["dreamlike"]
    },
    "Kandinsky 2.2": {
        "model_id": "kandinsky-community/kandinsky-2-2-decoder",
        "pipeline": KandinskyV22Pipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 8.0,
        "description": "Russian alternative to SD with unique artistic style",
        "size_gb": 4.5,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["kandinsky"]
    },
    "Pony Diffusion V6 XL": {
        "model_id": "LyliaEngine/Pony_Diffusion_V6_XL",
        "pipeline": StableDiffusionXLPipeline,
        "resolution_presets": SDXL_RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 8.0,
        "description": "Specialized model for stylized art",
        "size_gb": 7.0,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPTS["pony"]
    },
}

def is_model_downloaded(model_id):
    """Check if model is already in the cache"""
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                return True
        return False
    except:
        return False  # If we can't determine, assume it's not downloaded

class DownloadTracker(QThread):
    """Thread to check download progress"""
    progress = pyqtSignal(str)
    
    def __init__(self, model_id, size_gb):
        super().__init__()
        self.model_id = model_id
        self.size_gb = size_gb
        self.running = True
        
    def run(self):
        start_time = time.time()
        while self.running:
            elapsed = time.time() - start_time
            self.progress.emit(f"Downloading model {self.model_id} (~{self.size_gb:.1f}GB)... This may take several minutes.")
            if elapsed > 10:
                self.progress.emit(f"Still downloading {self.model_id}... (~{self.size_gb:.1f}GB, {elapsed:.0f}s elapsed)")
            
            # Allow the UI to process events for 2 seconds
            for _ in range(20):  # 20 * 0.1 seconds = 2 seconds
                if not self.running:
                    break
                time.sleep(0.1)
                
    def stop(self):
        self.running = False

class GenerationThread(QThread):
    finished = pyqtSignal(list)  # Changed to emit a list of images
    progress = pyqtSignal(int, str)  # Now includes status message
    image_ready = pyqtSignal(Image.Image, int, int)  # Emits (image, index, total)
    error = pyqtSignal(str)

    def __init__(self, model_config, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, seed=None, sampler=None, batch_size=1):
        super().__init__()
        self.model_config = model_config
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.seed = seed  # Random seed for generation (None = random)
        self.sampler = sampler  # Sampler algorithm to use
        self.batch_size = batch_size  # Number of images to generate
        self.pipe = None
        self.download_tracker = None
        self.generated_seeds = []  # Store seeds used for each image in the batch

    def progress_callback(self, step, timestep, latents):
        if step is not None:
            progress = int((step + 1) / self.num_inference_steps * 100)
            # Include the current image index in the batch in the status message
            if self.batch_size > 1:
                # We don't know which image we're generating in the callback, so use a class variable
                if not hasattr(self, 'current_image_index'):
                    self.current_image_index = 0
                status = f"Generating image {self.current_image_index + 1}/{self.batch_size}... Step {step + 1}/{self.num_inference_steps}"
            else:
                status = f"Generating image... Step {step + 1}/{self.num_inference_steps}"
            self.progress.emit(progress, status)

    def run(self):
        try:
            # Initialize the pipeline
            self.progress.emit(0, "Loading model...")
            model_id = self.model_config["model_id"]
            pipeline_class = self.model_config["pipeline"]
            size_gb = self.model_config.get("size_gb", 4.0)
            
            # Get the appropriate device and prepare
            device = get_device()
            
            # Clear CUDA cache if available
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    ErrorHandler.log_info(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
                except Exception as e:
                    ErrorHandler.log_warning(f"Unable to clear CUDA cache: {str(e)}")
            elif device == "mps":
                # No equivalent cache clearing for MPS
                pass
            
            # Check if model is already downloaded
            if not is_model_downloaded(model_id):
                self.progress.emit(0, f"First time using {model_id}. Downloading (~{size_gb:.1f}GB)...")
                
                # Start a thread to provide ongoing download status
                try:
                    self.download_tracker = DownloadTracker(model_id, size_gb)
                    self.download_tracker.progress.connect(lambda msg: self.progress.emit(0, msg))
                    self.download_tracker.start()
                    
                    # Give the tracker a moment to show its first message
                    time.sleep(0.5)
                    QApplication.processEvents()
                except Exception as e:
                    ErrorHandler.log_warning(f"Failed to start download tracker: {str(e)}")
            
            # Initialize pipeline
            ErrorHandler.log_info(f"Initializing pipeline for model: {model_id}")
            
            # Check if this is a local model
            is_local = self.model_config.get("is_local", False)
            
            # Get appropriate torch dtype
            torch_dtype = get_torch_dtype()
            
            if is_local and os.path.exists(model_id) and model_id.endswith((".safetensors", ".ckpt")):
                ErrorHandler.log_info(f"Loading local model from: {model_id}")
                self.progress.emit(0, f"Loading local model from: {os.path.basename(model_id)}...")
                
                try:
                    # Load from single file (used for Civitai models)
                    self.pipe = pipeline_class.from_single_file(
                        model_id,
                        torch_dtype=torch_dtype
                    )
                    ErrorHandler.log_info("Local model loaded successfully")
                except Exception as e:
                    error_msg = f"Error loading local model: {str(e)}"
                    ErrorHandler.log_warning(error_msg)
                    # Try the regular from_pretrained as a fallback
                    ErrorHandler.log_info("Trying alternative loading method...")
                    try:
                        self.pipe = pipeline_class.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            local_files_only=True
                        )
                    except Exception as fallback_error:
                        raise Exception(f"Failed to load local model using both methods. Original error: {str(e)}. Fallback error: {str(fallback_error)}")
                
                # Fix UNet for local SDXL models that expect additional embeddings
                try:
                    if pipeline_class == StableDiffusionXLPipeline or "XL" in model_id:
                        # Check if UNet has addition_embed_type set
                        if hasattr(self.pipe.unet, "config") and hasattr(self.pipe.unet.config, "addition_embed_type"):
                            if self.pipe.unet.config.addition_embed_type is not None:
                                ErrorHandler.log_info("Detected SDXL model that requires additional embeddings")
                                ErrorHandler.log_info("Setting addition_embed_type to None to prevent errors")
                                self.pipe.unet.config.addition_embed_type = None
                                ErrorHandler.log_info("UNet configuration updated for compatibility with local model")
                except Exception as e:
                    ErrorHandler.log_warning(f"Could not fix UNet configuration: {str(e)}")
                    ErrorHandler.log_warning("Generation might fail with this model")
            else:
                # Load remote model from Hugging Face Hub
                try:
                    self.pipe = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype
                    )
                except Exception as e:
                    if "401 Client Error" in str(e):
                        raise Exception(f"Authentication error: This model requires Hugging Face login. Please check your token. Error: {str(e)}")
                    elif "404 Client Error" in str(e):
                        raise Exception(f"Model not found: The model '{model_id}' does not exist or is not publicly available. Error: {str(e)}")
                    else:
                        raise Exception(f"Failed to load model '{model_id}': {str(e)}")
            
            # Special handling for Pony Diffusion - Set clip_skip=2
            if "pony" in model_id.lower() and hasattr(self.pipe, "text_encoder"):
                ErrorHandler.log_info("Setting clip_skip=2 for Pony Diffusion model")
                # For SDXL models with text_encoder and text_encoder_2
                if hasattr(self.pipe, "text_encoder_2"):
                    # Store only the penultimate layer's hidden states in both encoders
                    self.pipe.text_encoder.config.clip_skip = 2
                    self.pipe.text_encoder_2.config.clip_skip = 2
                else:
                    # For standard SD models with a single text encoder
                    self.pipe.text_encoder.config.clip_skip = 2
                ErrorHandler.log_info("Clip skip set to 2 for better results with Pony Diffusion")
            
            # Stop the download tracker if it's running
            if self.download_tracker:
                self.download_tracker.stop()
                self.download_tracker.wait()
                self.download_tracker = None
                
                # Let the user know download has completed
                self.progress.emit(0, f"Model {model_id} downloaded successfully! Moving to acceleration device...")
                QApplication.processEvents()  # Make sure the message is displayed
            
            # Move model to appropriate acceleration device
            if device != "cpu":
                self.progress.emit(0, f"Moving model to {device.upper()}...")
                ErrorHandler.log_info(f"Moving model to {device.upper()}...")
                try:
                    self.pipe = self.pipe.to(device)
                    
                    # Try to enable memory efficient attention if CUDA is available
                    if device == "cuda":
                        try:
                            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                                ErrorHandler.log_info("Enabling memory efficient attention...")
                                self.pipe.enable_xformers_memory_efficient_attention()
                        except Exception as e:
                            ErrorHandler.log_warning(f"Could not enable memory efficient attention: {str(e)}")
                            ErrorHandler.log_warning("Continuing without memory optimization...")
                except Exception as e:
                    if "CUDA out of memory" in str(e):
                        raise Exception(f"GPU out of memory while loading model. Try using CPU mode or a smaller model. Error: {str(e)}")
                    elif "MPS backend out of memory" in str(e):
                        raise Exception(f"Metal (MPS) out of memory while loading model. Try using CPU mode or a smaller model. Error: {str(e)}")
                    else:
                        raise Exception(f"Failed to move model to {device} device: {str(e)}")

            # Generate the images in batch
            self.progress.emit(0, f"Starting generation of {self.batch_size} images...")
            ErrorHandler.log_info(f"Starting generation with prompt: {self.prompt}")
            ErrorHandler.log_info(f"Negative prompt: {self.negative_prompt}")
            ErrorHandler.log_info(f"Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            ErrorHandler.log_info(f"Resolution: {self.width}x{self.height}, Batch size: {self.batch_size}")
            
            # Track seeds for each generated image
            self.generated_seeds = []
            generated_images = []
            
            for i in range(self.batch_size):
                # Update progress for each image in the batch
                self.progress.emit(int(i / self.batch_size * 100), f"Generating image {i+1} of {self.batch_size}...")
                
                # Set the current image index for the progress callback
                self.current_image_index = i
                
                # Set generator for reproducible results if seed is provided
                if i == 0 and self.seed is not None:
                    # For the first image, use the provided seed
                    generator = torch.Generator(device=get_device() if get_device() != "mps" else "cpu")
                    generator.manual_seed(self.seed)
                    current_seed = self.seed
                    ErrorHandler.log_info(f"Using seed for image {i+1}: {current_seed}")
                else:
                    # For subsequent images or if no seed was provided, use random seeds
                    generator = torch.Generator(device=get_device() if get_device() != "mps" else "cpu")
                    # Generate a seed that fits in a 32-bit integer to avoid overflow
                    current_seed = random.randint(0, 2147483647)
                    generator.manual_seed(current_seed)
                    ErrorHandler.log_info(f"Using random seed for image {i+1}: {current_seed}")
                
                # Store the seed for later reference
                self.generated_seeds.append(current_seed)
                
                # Set the sampler/scheduler if specified
                if self.sampler:
                    try:
                        # Map sampler names to scheduler classes
                        schedulers = {
                            "euler": EulerDiscreteScheduler,
                            "euler_ancestral": EulerAncestralDiscreteScheduler,
                            "heun": HeunDiscreteScheduler,
                            "dpm_2": KDPM2DiscreteScheduler,
                            "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
                            "lms": LMSDiscreteScheduler,
                            "pndm": PNDMScheduler,
                            "ddim": DDIMScheduler,
                            "ddpm": DDPMScheduler,
                            "deis": DEISMultistepScheduler,
                            "dpm_sde": DPMSolverSDEScheduler,
                            "dpm_solver": DPMSolverMultistepScheduler,
                            "karras_ve": KarrasVeScheduler,
                            "dpmsolver++": DPMSolverMultistepScheduler,
                            "dpmsingle": DPMSolverSinglestepScheduler,
                            "dpmpp_2m": DPMSolverMultistepScheduler,
                            "dpmpp_sde": DPMSolverSDEScheduler,
                            "dpmpp_2s_ancestral": DPMSolverSinglestepScheduler
                        }
                        
                        if self.sampler in schedulers:
                            ErrorHandler.log_info(f"Using sampler: {self.sampler}")
                            # Get the config from the current scheduler
                            old_config = self.pipe.scheduler.config
                            # Create new scheduler with the same config
                            new_scheduler = schedulers[self.sampler].from_config(old_config)
                            # Replace the scheduler
                            self.pipe.scheduler = new_scheduler
                        else:
                            ErrorHandler.log_warning(f"Unknown sampler: {self.sampler}, using default")
                    except Exception as e:
                        ErrorHandler.log_warning(f"Error setting sampler: {str(e)}")
                        ErrorHandler.log_warning("Using default sampler")
                
                # Generate the image with specific error handling for different error types
                with torch.inference_mode():
                    try:
                        # Different models might have slightly different APIs
                        if isinstance(self.pipe, StableDiffusionXLPipeline):
                            # For SDXL pipelines
                            image = self.pipe(
                                prompt=self.prompt,
                                negative_prompt=self.negative_prompt,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                width=self.width,
                                height=self.height,
                                callback=self.progress_callback,
                                callback_steps=1,
                                generator=generator
                            ).images[0]
                        elif isinstance(self.pipe, KandinskyV22Pipeline):
                            # Kandinsky has a different API
                            image = self.pipe(
                                prompt=self.prompt,
                                negative_prompt=self.negative_prompt,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                width=self.width,
                                height=self.height,
                                callback=self.progress_callback,
                                callback_steps=1,
                                generator=generator
                            ).images[0]
                        else:
                            # Standard StableDiffusion
                            image = self.pipe(
                                prompt=self.prompt,
                                negative_prompt=self.negative_prompt if self.model_config["supports_negative_prompt"] else None,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                width=self.width,
                                height=self.height,
                                callback=self.progress_callback,
                                callback_steps=1,
                                generator=generator
                            ).images[0]
                    except TypeError as e:
                        if "argument of type 'NoneType' is not iterable" in str(e):
                            print("Detected issue with added_cond_kwargs, trying with empty dict...")
                            if isinstance(self.pipe, StableDiffusionXLPipeline):
                                # Fallback for SDXL models with the NoneType error
                                image = self.pipe(
                                    prompt=self.prompt,
                                    negative_prompt=self.negative_prompt,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    width=self.width,
                                    height=self.height,
                                    callback=self.progress_callback,
                                    callback_steps=1,
                                    generator=generator,
                                    added_cond_kwargs={"text_embeds": torch.zeros(1, 1280, device=self.pipe.device), 
                                                      "time_ids": torch.zeros(1, 6, device=self.pipe.device)}
                                ).images[0]
                            else:
                                # For standard SD models with the same issue
                                image = self.pipe(
                                    prompt=self.prompt,
                                    negative_prompt=self.negative_prompt if self.model_config["supports_negative_prompt"] else None,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    width=self.width,
                                    height=self.height,
                                    callback=self.progress_callback,
                                    callback_steps=1,
                                    generator=generator,
                                    added_cond_kwargs={}
                                ).images[0]
                        else:
                            # Re-raise if it's a different error
                            raise
                
                generated_images.append(image)
                # Emit signal for UI update with each generated image
                self.image_ready.emit(image, i, self.batch_size)

            print(f"Generation of {self.batch_size} images completed successfully")
            self.finished.emit(generated_images)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
        finally:
            # Clean up
            if self.download_tracker:
                self.download_tracker.stop()
                self.download_tracker.wait()
            
            if self.pipe is not None:
                del self.pipe
            
            # Clean up device-specific resources
            if get_device() == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # No explicit cleanup needed for MPS devices

class MainWindow(QMainWindow):
    # Class variable to keep track of generation counter
    generation_counter = 0
    
    def __init__(self):
        super().__init__()
        
        # App setup
        self.setWindowTitle("DreamPixelForge - Text to Image")
        self.setGeometry(100, 100, 1000, 800)  # Slightly smaller default size
        
        # Create menu bar
        self.create_menu()
        
        # Create a scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)
        
        # Create the main content widget
        main_widget = QWidget()
        # Use QVBoxLayout for the main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(8)  # Reduced spacing for compactness
        main_layout.setContentsMargins(10, 10, 10, 10)  # Smaller margins
        
        # Set the main widget as the scroll area's widget
        scroll_area.setWidget(main_widget)
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient()
        self.ollama_available = self.ollama_client.is_available()
        self.ollama_models = []
        if self.ollama_available:
            self.ollama_models = self.ollama_client.list_models()
        self.ollama_thread = None
        
        # Create a split view - inputs on left, image on right for larger screens
        # Automatically switches to vertical on narrow screens
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        
        # Left panel - inputs and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)  # No margins for inner layout
        
        # Right panel - image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(0, 0, 0, 0)  # No margins for inner layout
        
        # Model selection section using CollapsibleSection
        model_section = CollapsibleSection("Model Selection")
        model_section.toggle_button.setChecked(True)  # Expanded by default
        model_section.on_toggle(True)  # Show content
        
        # Create a container for model selection content
        model_content = QWidget()
        model_layout = QVBoxLayout(model_content)
        model_layout.setSpacing(5)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a tabbed interface for model selection
        self.model_tabs = QTabWidget()
        # Make it very compact
        self.model_tabs.setMinimumHeight(70)
        self.model_tabs.setMaximumHeight(100)
        
        # Hugging Face models tab
        hf_tab = QWidget()
        hf_layout = QHBoxLayout(hf_tab)  # Changed to horizontal for compactness
        hf_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        hf_layout.setSpacing(5)  # Minimal spacing
        
        hf_model_label = QLabel("Select HuggingFace Model:")
        hf_model_label.setStyleSheet("font-weight: bold; color: white; font-size: 11px;")
        hf_model_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)  # Keep label size fixed
        
        # Use the right control based on platform
        if IS_MACOS:  # macOS
            self.model_combo = MacDropdownButton(list(AVAILABLE_MODELS.keys()))
            self.model_combo.currentTextChanged.connect(self.on_model_changed)
            # Set initial selection to Stable Diffusion 1.5
            self.model_combo.setCurrentText("Stable Diffusion 1.5")
        else:
            self.model_combo = QComboBox()
            self.model_combo.addItems(AVAILABLE_MODELS.keys())
            self.model_combo.currentTextChanged.connect(self.on_model_changed)
            # Set initial selection
            self.model_combo.setCurrentText("Stable Diffusion 1.5")
        
        hf_layout.addWidget(hf_model_label)
        hf_layout.addWidget(self.model_combo)
        
        # Local models tab
        local_tab = QWidget()
        local_layout = QHBoxLayout(local_tab)  # Changed to horizontal for compactness
        local_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        local_layout.setSpacing(5)  # Minimal spacing
        
        local_model_label = QLabel("Select Local Model:")
        local_model_label.setStyleSheet("font-weight: bold; color: white; font-size: 11px;")
        local_model_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)  # Keep label size fixed
        
        # Use the right control based on platform
        if IS_MACOS:  # macOS
            self.local_model_combo = MacDropdownButton()
            self.local_model_combo.currentTextChanged.connect(self.on_local_model_changed)
        else:
            self.local_model_combo = QComboBox()
            self.local_model_combo.currentTextChanged.connect(self.on_local_model_changed)
            
        manage_models_btn = QPushButton("Manage")
        manage_models_btn.setMaximumWidth(70)  # Limit width
        manage_models_btn.setStyleSheet("font-size: 11px; padding: 2px 5px;")
        manage_models_btn.clicked.connect(self.open_manage_models)
        
        local_layout.addWidget(local_model_label)
        local_layout.addWidget(self.local_model_combo, 1)  # Give dropdown stretch
        local_layout.addWidget(manage_models_btn)
        
        # Add tabs
        self.model_tabs.addTab(hf_tab, "Hugging Face Models")
        self.model_tabs.addTab(local_tab, "Local Models")
        
        # Connect tab change to update model info
        self.model_tabs.currentChanged.connect(self.update_model_info_from_tabs)
        
        # Add a model info label with compact styling
        self.model_info = QLabel(AVAILABLE_MODELS["Stable Diffusion 1.5"]["description"])
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("font-size: 11px; color: #aaa; margin-top: 0px;")
        self.model_info.setMaximumHeight(40)  # Limit height of model info
        
        model_layout.addWidget(self.model_tabs)
        model_layout.addWidget(self.model_info)
        
        # Add content to section
        model_section.addWidget(model_content)
        
        # Add section to left panel
        left_layout.addWidget(model_section)
        
        # Prompt section using CollapsibleSection
        prompt_section = CollapsibleSection("Prompt Settings")
        prompt_section.toggle_button.setChecked(True)  # Expanded by default
        prompt_section.on_toggle(True)  # Show content
        
        # Create a container for prompt content
        prompt_content = QWidget()
        prompt_layout = QVBoxLayout(prompt_content)
        prompt_layout.setSpacing(5)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        
        # Prompt input with label on top for compactness
        prompt_input_layout = QVBoxLayout()
        prompt_label = QLabel("Prompt:")
        prompt_label.setStyleSheet("font-weight: bold;")
        self.prompt_input = QLineEdit()
        prompt_input_layout.addWidget(prompt_label)
        prompt_input_layout.addWidget(self.prompt_input)
        
        # Negative prompt input with label on top for compactness
        neg_prompt_input_layout = QVBoxLayout()
        neg_prompt_label = QLabel("Negative Prompt:")
        neg_prompt_label.setStyleSheet("font-weight: bold;")
        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPTS["sd15"])
        neg_prompt_input_layout.addWidget(neg_prompt_label)
        neg_prompt_input_layout.addWidget(self.neg_prompt_input)
        
        prompt_layout.addLayout(prompt_input_layout)
        prompt_layout.addLayout(neg_prompt_input_layout)
        
        # Add content to section
        prompt_section.addWidget(prompt_content)
        
        # Add section to left panel
        left_layout.addWidget(prompt_section)
        
        # Ollama prompt enhancement section using the custom collapsible section
        if self.ollama_available and self.ollama_models:
            ollama_section = CollapsibleSection("Ollama Prompt Enhancement")
            
            # Create a container widget for Ollama controls
            ollama_content = QWidget()
            ollama_layout = QVBoxLayout(ollama_content)
            ollama_layout.setSpacing(5)
            ollama_layout.setContentsMargins(0, 0, 0, 0)
            
            # Model selection in a grid
            ollama_model_grid = QGridLayout()
            ollama_model_grid.setSpacing(5)
            
            ollama_model_label = QLabel("Ollama Model:")
            ollama_model_label.setStyleSheet("font-weight: bold; font-size: 11px;")
            
            # Use the right control based on platform
            if IS_MACOS:  # macOS
                self.ollama_model_combo = MacDropdownButton(self.ollama_models)
                self.ollama_model_combo.currentTextChanged.connect(self.refresh_ollama_models)
            else:
                self.ollama_model_combo = QComboBox()
                self.ollama_model_combo.addItems(self.ollama_models)
                self.ollama_model_combo.currentTextChanged.connect(self.refresh_ollama_models)
            
            refresh_ollama_button = QPushButton("Refresh")
            refresh_ollama_button.setMaximumWidth(70)
            
            refresh_ollama_button.clicked.connect(self.refresh_ollama_models)
            
            ollama_model_grid.addWidget(ollama_model_label, 0, 0)
            ollama_model_grid.addWidget(self.ollama_model_combo, 0, 1)
            ollama_model_grid.addWidget(refresh_ollama_button, 0, 2)
            
            # Input mode selection row
            input_mode_layout = QHBoxLayout()
            self.description_radio = QRadioButton("Description to Tags")
            self.tags_radio = QRadioButton("Enhance Tags")
            self.tags_radio.setChecked(True)  # Default to tag enhancement
            input_mode_layout.addWidget(self.description_radio)
            input_mode_layout.addWidget(self.tags_radio)
            
            # Enhance button and input row
            enhance_grid = QGridLayout()
            enhance_grid.setSpacing(5)
            
            self.enhance_input = QLineEdit()
            self.enhance_input.setPlaceholderText("Enter prompt to enhance")
            
            self.enhance_button = QPushButton("Enhance")
            self.enhance_button.setMaximumWidth(70)
            self.enhance_button.clicked.connect(self.enhance_prompt)
            
            enhance_grid.addWidget(self.enhance_input, 0, 0)
            enhance_grid.addWidget(self.enhance_button, 0, 1)
            
            # Add layouts to container
            ollama_layout.addLayout(ollama_model_grid)
            ollama_layout.addLayout(input_mode_layout)
            ollama_layout.addLayout(enhance_grid)
            
            # Add container to section
            ollama_section.addWidget(ollama_content)
            
            # Add section to left panel
            left_layout.addWidget(ollama_section)
        else:
            # Show a message when Ollama is not available
            ollama_section = CollapsibleSection("Ollama Prompt Enhancement")
            
            # Create message container
            ollama_content = QWidget()
            ollama_layout = QVBoxLayout(ollama_content)
            ollama_layout.setContentsMargins(0, 0, 0, 0)
            
            if not self.ollama_available:
                ollama_status = QLabel("Ollama is not running or not installed. Start Ollama to enable prompt enhancement.")
                ollama_status.setWordWrap(True)
            else:
                ollama_status = QLabel("No Ollama models found. Install models through Ollama to enable prompt enhancement.")
                ollama_status.setWordWrap(True)
                
            check_ollama_button = QPushButton("Check Ollama")
            check_ollama_button.clicked.connect(self.refresh_ollama_models)
            
            ollama_layout.addWidget(ollama_status)
            ollama_layout.addWidget(check_ollama_button)
            
            # Add container to section
            ollama_section.addWidget(ollama_content)
            
            # Add section to left panel
            left_layout.addWidget(ollama_section)
        
        # Parameters section using CollapsibleSection
        params_section = CollapsibleSection("Generation Parameters")
        params_section.toggle_button.setChecked(True)  # Expanded by default
        params_section.on_toggle(True)  # Show content
        
        # Create container for parameters content
        params_content = QWidget()
        params_layout = QGridLayout(params_content)
        params_layout.setSpacing(5)
        params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Batch size - Row 0
        batch_label = QLabel("Batch Size:")
        batch_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 10)  # Allow generating up to 10 images at once
        self.batch_input.setValue(1)  # Default to single image
        self.batch_input.setToolTip("Number of images to generate in one batch")
        params_layout.addWidget(batch_label, 0, 0)
        params_layout.addWidget(self.batch_input, 0, 1)
        
        # Resolution selection - Row 0 (continued)
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        # Create the appropriate control based on platform
        if IS_MACOS:  # macOS
            self.resolution_combo = MacDropdownButton()
            default_model = list(AVAILABLE_MODELS.keys())[0]
            default_resolutions = AVAILABLE_MODELS[default_model].get('resolution_presets', {}).keys()
            self.resolution_combo.addItems(default_resolutions)
            # Set the initial text to show the selected option
            if default_resolutions:
                first_resolution = list(default_resolutions)[0]
                self.resolution_combo.setText(f"{first_resolution} ▼")
        else:
            self.resolution_combo = QComboBox()
            default_model = list(AVAILABLE_MODELS.keys())[0]
            default_resolutions = AVAILABLE_MODELS[default_model].get('resolution_presets', {}).keys()
            self.resolution_combo.addItems(default_resolutions)
        
        params_layout.addWidget(resolution_label, 0, 2)
        params_layout.addWidget(self.resolution_combo, 0, 3)
        
        # Number of steps - Row 1
        steps_label = QLabel("Steps:")
        steps_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 150)
        self.steps_input.setValue(30)
        params_layout.addWidget(steps_label, 1, 0)
        params_layout.addWidget(self.steps_input, 1, 1)
        
        # Guidance scale - Row 1 (continued)
        guidance_label = QLabel("Guidance Scale:")
        guidance_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.guidance_input = QDoubleSpinBox()
        self.guidance_input.setRange(1.0, 20.0)
        self.guidance_input.setValue(7.5)
        self.guidance_input.setSingleStep(0.5)
        params_layout.addWidget(guidance_label, 1, 2)
        params_layout.addWidget(self.guidance_input, 1, 3)
        
        # Seed - Row 2
        seed_label = QLabel("Seed:")
        seed_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.seed_input = QLineEdit("-1")
        self.seed_input.setPlaceholderText("Enter seed or -1 for random")
        random_seed_button = QPushButton("Random")
        random_seed_button.setMaximumWidth(70)
        random_seed_button.clicked.connect(self.generate_random_seed)
        
        params_layout.addWidget(seed_label, 2, 0)
        params_layout.addWidget(self.seed_input, 2, 1)
        params_layout.addWidget(random_seed_button, 2, 2)
        
        # Sampler selection - Row 3
        sampler_label = QLabel("Sampler:")
        sampler_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        
        # Create the appropriate control based on platform
        if IS_MACOS:  # macOS
            self.sampler_combo = MacDropdownButton(SAMPLERS.keys())
            # Set the initial text to show the selected option
            sampler_list = list(SAMPLERS.keys())
            if sampler_list:
                self.sampler_combo.setText(f"{sampler_list[0]} ▼")
        else:
            self.sampler_combo = QComboBox()
            self.sampler_combo.addItems(SAMPLERS.keys())
        
        params_layout.addWidget(sampler_label, 3, 0)
        params_layout.addWidget(self.sampler_combo, 3, 1, 1, 3)  # Span 3 columns
        
        # Add content to section
        params_section.addWidget(params_content)
        
        # Add param section to left panel
        left_layout.addWidget(params_section)
        
        # Generate button
        self.generate_button = QPushButton("Generate Images")
        self.generate_button.setMinimumHeight(36)  # Slightly taller for emphasis
        self.generate_button.clicked.connect(self.generate_image)
        left_layout.addWidget(self.generate_button)
        
        # Status and progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        left_layout.addLayout(status_layout)
        
        # Add stretch to push everything up
        left_layout.addStretch(1)
        
        # Right panel - image display
        # Image display with proper sizing
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(384, 384)  # Smaller minimum size
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid #666; background-color: #2a2a2a;")
        
        # Navigation controls (will be added when needed)
        self.nav_placeholder = QWidget()
        self.nav_layout = QHBoxLayout(self.nav_placeholder)
        self.nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # Save button
        self.save_button = QPushButton("Save Image to Custom Location")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        
        right_layout.addWidget(self.image_label, 1)  # Give it stretch factor
        right_layout.addWidget(self.nav_placeholder)
        right_layout.addWidget(self.save_button)
        
        # Add panels to splitter
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        
        # Set initial sizes - 40% for controls, 60% for image
        self.main_splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        self.current_images = []
        self.generation_thread = None
        
        # Initialize local models dropdown with auto-imported models
        self.refresh_local_models()
        
        # Create outputs directory if it doesn't exist
        self.outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
            print(f"Created outputs directory at {self.outputs_dir}")
        
        # Initialize the counter by finding the highest existing count
        self.initialize_counter()
        
        # Check for first use
        self.check_first_use()
        
        # Initialize with the default model
        default_model = "Stable Diffusion 1.5"
        self.on_model_changed(default_model)
        
        # Platform-specific styling for macOS
        if IS_MACOS:  # macOS
            # Create a comprehensive macOS style sheet
            mac_style = """
                /* Global styles */
                QWidget {
                    font-size: 13px;
                }
                
                /* Line edit styling */
                QLineEdit {
                    padding: 4px;
                    min-height: 22px;
                    border: 1px solid #777;
                    border-radius: 3px;
                    background-color: #333;
                }
                
                /* Button styling */
                QPushButton {
                    padding: 4px 8px;
                    min-height: 22px;
                    border: 1px solid #777;
                    border-radius: 3px;
                    background-color: #444;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                
                /* Group box styling */
                QGroupBox {
                    border: 1px solid #666;
                    border-radius: 3px;
                    margin-top: 0.5em;
                    padding: 2px;
                    background-color: #333;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 7px;
                    top: -2px;
                    padding: 0 3px;
                    background-color: #333;
                }
                QGroupBox::indicator {
                    width: 13px;
                    height: 13px;
                }
                
                /* Tab widget styling */
                QTabWidget::pane {
                    border: 1px solid #777;
                    border-radius: 3px;
                    top: -1px;
                    padding: 2px;
                }
                QTabBar::tab {
                    font-size: 12px;
                    padding: 4px 8px;
                    min-width: 100px;
                    background-color: #333;
                    border: 1px solid #555;
                    border-bottom: none;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #444;
                    border: 1px solid #777;
                    border-bottom: none;
                }
                
                /* Spin box styling */
                QSpinBox, QDoubleSpinBox {
                    padding: 2px;
                    min-height: 22px;
                    border: 1px solid #777;
                    border-radius: 3px;
                    background-color: #333;
                }
                
                /* Progress bar styling */
                QProgressBar {
                    border: 1px solid #666;
                    border-radius: 3px;
                    background-color: #333;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #5a8cbe;
                    width: 10px;
                }
                
                /* Splitter styling */
                QSplitter::handle {
                    background-color: #555;
                }
                QSplitter::handle:horizontal {
                    width: 4px;
                }
                QSplitter::handle:vertical {
                    height: 4px;
                }
                
                /* ScrollArea styling */
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
            """
            
            # Apply specific styles to critical elements
            self.setStyleSheet(mac_style)
            
            # No need for the old layout adjustments as we're using a new layout system
            # The spacing is already set in the layout initialization

    def initialize_counter(self):
        """Find the highest counter value from existing files in the outputs directory"""
        try:
            if not os.path.exists(self.outputs_dir):
                return
                
            highest_counter = 0
            # Search for files matching the pattern: ####_*.png
            for filename in os.listdir(self.outputs_dir):
                if filename.endswith(".png") and "_" in filename:
                    # Try to extract the counter value from the filename
                    try:
                        counter_str = filename.split("_")[0]
                        counter = int(counter_str)
                        highest_counter = max(highest_counter, counter)
                    except (ValueError, IndexError):
                        # If file doesn't match our expected format, skip it
                        continue
            
            # Set the class counter to the highest found + 1
            if highest_counter > 0:
                MainWindow.generation_counter = highest_counter
                print(f"Initialized counter to {MainWindow.generation_counter} based on existing files")
        except Exception as e:
            print(f"Error initializing counter: {str(e)}")

    def create_menu(self):
        """Create the menu bar"""
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # Import model action
        import_model_action = QAction("Import Model", self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        
        # Manage models action
        manage_models_action = QAction("Manage Models", self)
        manage_models_action.triggered.connect(self.open_manage_models)
        file_menu.addAction(manage_models_action)
        
        # Separator
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Models menu
        models_menu = menu_bar.addMenu("Models")
        
        # Open models folder action
        open_folder_action = QAction("Open Models Folder", self)
        open_folder_action.triggered.connect(self.open_models_folder)
        models_menu.addAction(open_folder_action)
        
        # Presets menu
        presets_menu = menu_bar.addMenu("Presets")
        
        # App Icon Generator action
        app_icon_action = QAction("App Icon Generator", self)
        app_icon_action.triggered.connect(self.apply_app_icon_preset)
        presets_menu.addAction(app_icon_action)
        
        # Post Processing menu
        post_processing_menu = menu_bar.addMenu("Post Processing")
        
        # App Icon Post-Processing action
        post_process_action = QAction("App Icon Processing", self)
        post_process_action.triggered.connect(self.post_process_app_icon)
        post_processing_menu.addAction(post_process_action)
    
    def import_model(self):
        """Open dialog to import a model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model File", 
            "", 
            "Model Files (*.safetensors *.ckpt)"
        )
        
        if file_path:
            dialog = AddLocalModelDialog(self, file_path)
            if dialog.exec():
                model_info = dialog.get_model_info()
                if model_info:
                    LOCAL_MODELS[model_info.name] = model_info
                    self.refresh_local_models()
                    
                    # Switch to the local models tab
                    self.model_tabs.setCurrentIndex(1)
                    
                    # Select the newly added model
                    index = self.local_model_combo.findText(model_info.name)
                    if index >= 0:
                        self.local_model_combo.setCurrentIndex(index)
    
    def open_models_folder(self):
        """Open the models folder in the file explorer"""
        import subprocess
        import os
        
        # Create the folder if it doesn't exist
        os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
        
        # Open the folder in the default file explorer
        if sys.platform == "win32":
            os.startfile(LOCAL_MODELS_DIR)
        elif sys.platform == "darwin":
            subprocess.call(["open", LOCAL_MODELS_DIR])
        else:
            subprocess.call(["xdg-open", LOCAL_MODELS_DIR])

    def check_first_use(self):
        """Show a message if this is the first use about model downloads"""
        model_id = AVAILABLE_MODELS["Stable Diffusion 1.5"]["model_id"]
        if not is_model_downloaded(model_id):
            QMessageBox.information(
                self,
                "First Time Use",
                "The first time you use each model, it will be downloaded from Hugging Face.\n\n"
                "This can take several minutes depending on your internet speed.\n\n"
                "- Stable Diffusion models: ~4GB each\n"
                "- Stable Diffusion XL: ~6.5GB\n"
                "- Pony Diffusion V6 XL: ~7GB\n\n"
                "Models are downloaded only once and cached for future use."
            )

    def on_model_changed(self, model_name):
        """Update UI when model selection changes"""
        # Cancel any running generation thread
        self.stop_generation_if_running()
        
        # Make sure the combo box display is updated properly (especially important on macOS)
        if IS_MACOS:
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        
        model_config = AVAILABLE_MODELS[model_name]
        self.model_info.setText(model_config["description"])
        self.guidance_input.setValue(model_config["default_guidance_scale"])
        self.update_resolutions()
        
        # Enable/disable negative prompt based on model support
        self.neg_prompt_input.setEnabled(model_config["supports_negative_prompt"])
        
        # Set the appropriate negative prompt for this model
        if "negative_prompt" in model_config and model_config["supports_negative_prompt"]:
            self.neg_prompt_input.setText(model_config["negative_prompt"])
        
        # Show download size information if model isn't downloaded yet
        if not is_model_downloaded(model_config["model_id"]):
            size_gb = model_config.get("size_gb", 4.0)
            self.status_label.setText(f"Note: {model_name} (~{size_gb:.1f}GB) will be downloaded on first use")
        else:
            # Reset status label if model is already downloaded
            self.status_label.setText("Ready")
            
    def stop_generation_if_running(self):
        """Stop any running generation thread"""
        if hasattr(self, 'generation_thread') and self.generation_thread and self.generation_thread.isRunning():
            print("Stopping current generation due to model change")
            
            # Disconnect signals to prevent further UI updates
            try:
                self.generation_thread.progress.disconnect()
                self.generation_thread.image_ready.disconnect()
                self.generation_thread.error.disconnect()
                self.generation_thread.finished.disconnect()
            except:
                # It's fine if signals were already disconnected
                pass
            
            # Terminate the thread
            self.generation_thread.terminate()
            self.generation_thread.wait(100)  # Wait briefly for termination
            
            # Update UI to reflect cancellation
            try:
                if hasattr(self, 'generate_button'):
                    self.generate_button.setEnabled(True)
                
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.setVisible(False)
                
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Generation cancelled due to model change")
            except RuntimeError:
                # UI elements were likely deleted during model change
                print("UI update error: Some UI elements have been deleted")

    def update_resolutions(self):
        """Update the resolution presets based on selected model"""
        current_model = self.model_combo.currentText()
        model_config = AVAILABLE_MODELS[current_model]
        self.update_resolutions_from_config(model_config)

    def generate_random_seed(self):
        """Generate a random seed value"""
        random_seed = random.randint(0, 2147483647)
        self.seed_input.setText(str(random_seed))
        
    def generate_image(self):
        if not self.prompt_input.text():
            self.status_label.setText("Please enter a prompt before generating images")
            return
            
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        
        # Get current model configuration based on active tab
        current_tab = self.model_tabs.currentIndex()
        if current_tab == 0:  # Hugging Face tab
            current_model = self.model_combo.currentText()
            model_config = AVAILABLE_MODELS[current_model]
        else:  # Local models tab
            current_model = self.local_model_combo.currentText()
            if not current_model or current_model not in LOCAL_MODELS:
                self.status_label.setText("No local model selected.")
                self.generate_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
            
            model_config = LOCAL_MODELS[current_model].get_config()
        
        # Get selected resolution
        selected_resolution = self.resolution_combo.currentText()
        resolution_presets = model_config["resolution_presets"]
        
        # Check if the selected resolution exists in the presets
        if selected_resolution in resolution_presets:
            width, height = resolution_presets[selected_resolution]
        else:
            # Fallback to the first resolution preset if the selected one isn't available
            first_preset = list(resolution_presets.keys())[0]
            width, height = resolution_presets[first_preset]
            print(f"Selected resolution '{selected_resolution}' not found in model presets. Using '{first_preset}' instead.")
        
        # Process seed value (-1 means random/None)
        seed_value = self.seed_input.text()
        if seed_value == "-1":
            seed_value = None
            self.status_label.setText("Generating with random seed...")
        else:
            try:
                seed_value = int(seed_value)
                self.status_label.setText(f"Generating with seed: {seed_value}...")
            except ValueError:
                QMessageBox.warning(self, "Invalid Seed Format", "Please enter a valid integer seed or -1 for a random seed.")
                self.generate_button.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
        
        # Get selected sampler
        sampler_name = self.sampler_combo.currentText()
        sampler_id = SAMPLERS.get(sampler_name)
        
        # Special prompt handling for Pony Diffusion
        prompt = self.prompt_input.text()
        negative_prompt = self.neg_prompt_input.text()
        
        # Check if this is a Pony model (either HuggingFace or local)
        is_pony_model = False
        if current_tab == 0 and "pony diffusion" in current_model.lower():
            is_pony_model = True
        elif current_tab == 1 and current_model in LOCAL_MODELS:
            # Check if it's a locally imported pony model
            if hasattr(LOCAL_MODELS[current_model], 'pony_override') and LOCAL_MODELS[current_model].pony_override:
                is_pony_model = True
            # Also check the filename for pony indicators
            elif "pony" in LOCAL_MODELS[current_model].file_path.lower():
                is_pony_model = True
        
        # Add quality boosters for Pony models
        if is_pony_model:
            # Check if quality boosters are already in the prompt
            if not any(booster in prompt.lower() for booster in ["score_9", "score_8_up"]):
                # Add quality boosters as recommended for Pony Diffusion, using only the first 3 as requested
                quality_booster = "score_9, score_8_up, score_7_up"
                if prompt.strip():
                    prompt = f"{quality_booster}, {prompt}"
                else:
                    prompt = quality_booster
                self.status_label.setText("Added quality boosters for Pony model")
        
        # Reset current images when starting a new generation
        batch_size = self.batch_input.value()
        self.current_images = [None] * batch_size
        self.current_image_index = None  # Will be set when the first image is ready
        
        # Remove previous image navigation if it exists
        if hasattr(self, 'image_nav_layout'):
            self._remove_image_navigation()
            
        # Create and start the generation thread
        try:
            self.generation_thread = GenerationThread(
                model_config=model_config,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.steps_input.value(),
                guidance_scale=self.guidance_input.value(),
                width=width,
                height=height,
                seed=seed_value,
                sampler=sampler_id,
                batch_size=batch_size
            )
            
            # Connect signals
            self.generation_thread.progress.connect(self.handle_progress)
            self.generation_thread.image_ready.connect(self.handle_image_ready)
            self.generation_thread.error.connect(self.handle_error)
            self.generation_thread.finished.connect(self.handle_generation_finished)
            
            # Start the thread
            self.generation_thread.start()
        except Exception as e:
            print(f"Error starting generation: {str(e)}")
            self.status_label.setText("Error: Failed to start generation")
            self.generate_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def handle_progress(self, progress, message):
        """Handle progress updates from generation thread"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        
    def handle_error(self, error_message):
        """Handle errors from generation thread with improved error handling"""
        self.status_label.setText("Error: Generation failed")
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Use the centralized error handler
        ErrorHandler.handle_generation_error(self, error_message, "Generation Error")

    def handle_generation_finished(self, images):
        """Handle completion of generation thread"""
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation complete! All images auto-saved to outputs folder.")
        
    def handle_image_ready(self, image, index, total):
        """Handle individual image ready signal from generation thread"""
        def _ui_update_operation():
            # Store the image in the current_images list
            # Make sure the list is large enough
            while len(self.current_images) < total:
                self.current_images.append(None)
                
            # Update the image at this index
            self.current_images[index] = image
            
            # Log the image generation
            # Get the seed from the generation thread
            seed = self.generation_thread.generated_seeds[index] if hasattr(self.generation_thread, 'generated_seeds') and len(self.generation_thread.generated_seeds) > index else -1
            image_path = self.auto_save_image(image, index, seed)
            if image_path:
                ErrorHandler.log_info(f"Automatically saved image {index+1}/{total} to {image_path}")
                
            # Display the image
            self.display_image(index)
            
            # Update status
            self.status_label.setText(f"Generated image {index+1}/{total}")
            
            # Only show image navigation for multiple images
            if total > 1:
                self._setup_image_navigation(index, total)
        
        # Use safe UI operation with error handling
        ErrorHandler.safe_ui_operation(self, _ui_update_operation, "Image Ready Error")

    def display_image(self, index):
        """Display an image at the specified index with proper error handling"""
        if not self.current_images or index < 0 or index >= len(self.current_images):
            return
            
        def _display_operation():
            try:
                # Get the image and convert to QPixmap
                pil_image = self.current_images[index]
                
                # Convert PIL image to QPixmap
                qim = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qim)
                
                # Calculate the appropriate size to display the image while maintaining aspect ratio
                label_size = self.image_label.size()
                scaled_pixmap = pixmap.scaled(
                    label_size.width(), 
                    label_size.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Set the pixmap to the label
                self.image_label.setPixmap(scaled_pixmap)
                
                # Enable save button since we have an image
                self.save_button.setEnabled(True)
                
                # Setup image navigation if we have multiple images
                self._setup_image_navigation(index, len(self.current_images))
                
                # Update window title with current image info
                image_info = f"Image {index+1}/{len(self.current_images)}"
                self.setWindowTitle(f"DreamPixelForge - Text to Image - {image_info}")
                
                # When the image label is resized, make sure the image gets resized too
                if not hasattr(self.image_label, 'resizeEvent_original'):
                    self.image_label.resizeEvent_original = self.image_label.resizeEvent
                    
                    def new_resize_event(event):
                        # Call the original resize event handler
                        self.image_label.resizeEvent_original(event)
                        # Re-display the current image to adjust scaling
                        if self.current_images and index < len(self.current_images):
                            # Convert PIL image to QPixmap
                            current_image = self.current_images[index]
                            qim = QImage(current_image.tobytes(), current_image.width, current_image.height, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(qim)
                            
                            # Scale to the new size
                            label_size = self.image_label.size()
                            scaled_pixmap = pixmap.scaled(
                                label_size.width(), 
                                label_size.height(),
                                Qt.AspectRatioMode.KeepAspectRatio, 
                                Qt.TransformationMode.SmoothTransformation
                            )
                            
                            # Set the pixmap to the label
                            self.image_label.setPixmap(scaled_pixmap)
                    
                    # Replace the resize event with our custom one
                    self.image_label.resizeEvent = new_resize_event
                
            except Exception as e:
                ErrorHandler.log_error(f"Error displaying image at index {index}: {str(e)}", exc_info=e)
        
        # Use the safe_ui_operation method to handle any exceptions
        ErrorHandler.safe_ui_operation(self, _display_operation, "Display Image Error")

    def show_previous_image(self):
        """Show the previous image in the batch"""
        try:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.display_image(self.current_image_index)
                
                # Update navigation buttons
                if hasattr(self, 'image_counter_label'):
                    self.image_counter_label.setText(f"Image {self.current_image_index+1}/{len(self.current_images)}")
                    self.prev_button.setEnabled(self.current_image_index > 0)
                    self.next_button.setEnabled(self.current_image_index < len(self.current_images) - 1)
        except Exception as e:
            ErrorHandler.handle_ui_error(self, e, "Navigation Error")

    def show_next_image(self):
        """Show the next image in the batch"""
        try:
            if self.current_image_index < len(self.current_images) - 1:
                self.current_image_index += 1
                self.display_image(self.current_image_index)
                
                # Update navigation buttons
                if hasattr(self, 'image_counter_label'):
                    self.image_counter_label.setText(f"Image {self.current_image_index+1}/{len(self.current_images)}")
                    self.prev_button.setEnabled(self.current_image_index > 0)
                    self.next_button.setEnabled(self.current_image_index < len(self.current_images) - 1)
        except Exception as e:
            ErrorHandler.handle_ui_error(self, e, "Navigation Error")
            
    def _setup_image_navigation(self, index, total):
        """Setup image navigation UI with proper error handling"""
        if total <= 1:
            # Remove navigation if it exists
            if hasattr(self, 'image_nav_layout') and hasattr(self, 'nav_placeholder'):
                try:
                    # Clear existing items from nav layout
                    while self.nav_layout.count():
                        item = self.nav_layout.takeAt(0)
                        widget = item.widget()
                        if widget:
                            widget.deleteLater()
                except RuntimeError:
                    ErrorHandler.log_warning("Navigation widgets have been deleted, cannot update")
            return
            
        # Update existing navigation if it exists
        if hasattr(self, 'image_counter_label'):
            try:
                self.image_counter_label.setText(f"Image {index+1}/{total}")
                self.prev_button.setEnabled(index > 0)
                self.next_button.setEnabled(index < total - 1)
                return
            except RuntimeError:
                ErrorHandler.log_warning("Navigation widgets have been deleted, cannot create new ones")
                
        # Create new navigation for multiple images
        try:
            # Create navigation buttons for browsing images
            self.prev_button = QPushButton("Previous")
            self.prev_button.clicked.connect(self.show_previous_image)
            self.prev_button.setEnabled(False)  # Disabled at first image
            
            self.image_counter_label = QLabel(f"Image {index+1}/{total}")
            self.image_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.next_button = QPushButton("Next")
            self.next_button.clicked.connect(self.show_next_image)
            self.next_button.setEnabled(index < total - 1)  # Only enabled if there are more images
            
            # Add widgets to the navigation layout
            self.nav_layout.addWidget(self.prev_button)
            self.nav_layout.addWidget(self.image_counter_label)
            self.nav_layout.addWidget(self.next_button)
            
            # Make sure the nav_placeholder is visible
            self.nav_placeholder.setVisible(True)
            
            ErrorHandler.log_info(f"Created image navigation for {total} images, current index {index}")
        except Exception as e:
            ErrorHandler.log_error(f"Failed to create image navigation: {str(e)}", exc_info=e)

    def _insert_navigation_layout(self):
        """This method is kept for compatibility with the old code but is not needed with the new layout"""
        pass  # Navigation is now added directly to the right panel in __init__
        
    def _remove_image_navigation(self):
        """Remove navigation buttons"""
        if hasattr(self, 'nav_placeholder'):
            try:
                # Clear the navigation layout
                while self.nav_layout.count():
                    item = self.nav_layout.takeAt(0)
                    widget = item.widget()
                    if widget:
                        widget.deleteLater()
                
                # Hide the placeholder when not needed
                self.nav_placeholder.setVisible(False)
                
                ErrorHandler.log_info("Image navigation removed")
            except Exception as e:
                ErrorHandler.log_error(f"Failed to remove image navigation: {str(e)}", exc_info=e)

    def on_local_model_changed(self, model_name):
        """Handle change in local model selection"""
        # Cancel any running generation thread
        self.stop_generation_if_running()
        
        if not model_name or model_name not in LOCAL_MODELS:
            return

        # Make sure the combo box display is updated properly (especially important on macOS)
        if IS_MACOS:
            index = self.local_model_combo.findText(model_name)
            if index >= 0:
                self.local_model_combo.setCurrentIndex(index)

        model_info = LOCAL_MODELS[model_name]
        config = model_info.get_config()
        
        self.model_info.setText(config["description"])
        self.guidance_input.setValue(config["default_guidance_scale"])
        
        # Update resolution presets based on model type
        self.update_resolutions_from_config(config)
        
        # Enable/disable negative prompt based on model support
        self.neg_prompt_input.setEnabled(config["supports_negative_prompt"])
        
        # Set the negative prompt based on the model config
        if config["supports_negative_prompt"] and "negative_prompt" in config:
            self.neg_prompt_input.setText(config["negative_prompt"])
        # Fall back to determining based on model type if not in config
        elif config["supports_negative_prompt"]:
            # Determine which negative prompt to use based on model type
            if isinstance(config["pipeline"], StableDiffusionXLPipeline):
                self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPTS["sdxl"])
            elif model_info.model_type == "Stable Diffusion 2.1":
                self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPTS["sd21"])
            else:
                # Default to SD 1.5 negative prompt for other model types
                self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPTS["sd15"])

    def update_resolutions_from_config(self, model_config):
        """Update the resolution presets based on provided configuration"""
        resolution_presets = model_config["resolution_presets"]
        
        # Remember current selection if possible
        current_selection = self.resolution_combo.currentText()
        
        self.resolution_combo.clear()
        
        # Handle adding items based on platform
        if IS_MACOS:
            # Add items and update display for Mac dropdown
            self.resolution_combo.addItems(resolution_presets.keys())
            
            # Try to restore previous selection if it exists in the new list
            index = self.resolution_combo.findText(current_selection)
            if index >= 0:
                self.resolution_combo.setCurrentIndex(index)
            elif len(resolution_presets) > 0:
                # Set the first item if previous selection not found
                first_resolution = list(resolution_presets.keys())[0]
                self.resolution_combo.setText(f"{first_resolution} ▼")
        else:
            # Standard combo box for Windows/Linux
            self.resolution_combo.addItems(resolution_presets.keys())
            
            # Try to restore previous selection if it exists in the new list
            index = self.resolution_combo.findText(current_selection)
            if index >= 0:
                self.resolution_combo.setCurrentIndex(index)

    def open_manage_models(self):
        """Open the local models dialog"""
        dialog = LocalModelsDialog(self)
        dialog.models_updated.connect(self.refresh_local_models)
        dialog.exec()

    def refresh_local_models(self):
        """Refresh the list of local models in the UI"""
        selected_model = self.local_model_combo.currentText()
        
        self.local_model_combo.clear()
        
        if LOCAL_MODELS:
            self.local_model_combo.addItems(LOCAL_MODELS.keys())
            
            # Try to restore previous selection
            if selected_model in LOCAL_MODELS:
                index = self.local_model_combo.findText(selected_model)
                if index >= 0:
                    self.local_model_combo.setCurrentIndex(index)
            
            # Update the current model info
            if self.local_model_combo.currentText():
                self.on_local_model_changed(self.local_model_combo.currentText())
        else:
            self.model_info.setText("No local models available. Use 'Manage Models' to add one.")
            
    def refresh_ollama_models(self):
        # Fetch available models from Ollama
        self.ollama_models = []
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                self.ollama_models = [model['name'] for model in models_data['models']]
            else:
                QMessageBox.warning(
                    self, "Connection Error", 
                    f"Failed to connect to Ollama API. Status code: {response.status_code}"
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Connection Error", 
                f"Failed to connect to Ollama API: {str(e)}"
            )
        
        # Update the model dropdown
        if IS_MACOS:  # macOS
            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItems(self.ollama_models)
        else:
            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItems(self.ollama_models)
    
    def update_model_info_from_tabs(self):
        """Update model info when switching between Hugging Face and Local models tabs"""
        current_tab = self.model_tabs.currentIndex()
        if current_tab == 0:  # Hugging Face tab
            if self.model_combo.currentText():  # If there's a selected model
                self.on_model_changed(self.model_combo.currentText())
        else:  # Local models tab
            if self.local_model_combo.currentText():  # If there's a selected model
                self.on_local_model_changed(self.local_model_combo.currentText())

    def enhance_prompt(self):
        """Use Ollama to enhance the prompt"""
        if not self.enhance_input.text():
            return
            
        # Disable the enhance button while processing
        self.enhance_button.setEnabled(False)
        self.status_label.setText("Enhancing prompt with Ollama...")
        
        # Get the selected model and mode
        if IS_MACOS:  # macOS
            model = self.ollama_model_combo.currentText()
        else:
            model = self.ollama_model_combo.currentText()
            
        mode = "description" if self.description_radio.isChecked() else "tags"
        
        # Create and start the Ollama thread
        self.ollama_thread = OllamaThread(
            self.ollama_client,
            model,
            self.enhance_input.text(),
            mode
        )
        
        self.ollama_thread.finished.connect(self.handle_enhanced_prompt)
        self.ollama_thread.error.connect(self.handle_ollama_error)
        self.ollama_thread.start()
    
    def handle_enhanced_prompt(self, enhanced_prompt):
        """Handle the enhanced prompt from Ollama"""
        self.enhance_button.setEnabled(True)
        self.status_label.setText("Prompt enhanced successfully")
        
        # Set the enhanced prompt as the main prompt
        self.prompt_input.setText(enhanced_prompt)
        
        # Clear the enhance input
        self.enhance_input.clear()
    
    def handle_ollama_error(self, error_message):
        """Handle errors from Ollama"""
        self.status_label.setText(f"Ollama error: {error_message}")
        
        # Re-enable the enhance button if it was disabled
        if hasattr(self, 'enhance_button'):
            self.enhance_button.setEnabled(True)
            
        # Show error message if really needed
        QMessageBox.warning(self, "Ollama Error", error_message)
    
    def apply_app_icon_preset(self):
        """Apply a preset for generating app icons
        
        This preset applies optimal settings for app icon generation based on best practices
        for creating clean, professional app icons with Stable Diffusion.
        """
        # Get the current model tab
        current_tab = self.model_tabs.currentIndex()
        model_name = ""
        
        # Check if we're using a model that supports SDXL-quality generations
        is_sdxl_compatible = False
        if current_tab == 0:  # Hugging Face tab
            model_name = self.model_combo.currentText()
            is_sdxl_compatible = "xl" in model_name.lower() or "sdxl" in model_name.lower()
        else:  # Local models tab
            model_name = self.local_model_combo.currentText()
            if model_name and model_name in LOCAL_MODELS:
                is_sdxl_compatible = "Stable Diffusion XL" in LOCAL_MODELS[model_name].model_type
        
        # Select the appropriate square resolution for icons
        if is_sdxl_compatible:
            square_resolution = "1024x1024 (Square)"
        else:
            square_resolution = "512x512 (Square)"
            
        # Find and set the resolution
        index = self.resolution_combo.findText(square_resolution)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)
        
        # Set optimal inference steps (20-30 is good for detailed icons)
        self.steps_input.setValue(25)
        
        # Set optimal guidance scale (7.0-7.5 is good for icons - balances creativity and prompt adherence)
        self.guidance_input.setValue(7.0)
        
        # Set batch size to 4 for options
        self.batch_input.setValue(4)
        
        # Set to random seed for variety
        self.seed_input.setText("-1")
        
        # Set optimal sampler for detailed images
        sampler_index = self.sampler_combo.findText("DPM++ 2M")
        if sampler_index < 0:
            sampler_index = self.sampler_combo.findText("DDIM")  # Fallback
        
        if sampler_index >= 0:
            self.sampler_combo.setCurrentIndex(sampler_index)
            
        # Create an app icon specific negative prompt
        negative_prompt = "ugly, blurry, poor quality, distorted, deformed, poorly drawn, bad anatomy, " + \
                         "text, words, letters, signature, watermark, logo, duplicated, extra details, " + \
                         "cluttered, noisy, busy design, low contrast, pixelated"
        
        # Update the negative prompt
        self.neg_prompt_input.setText(negative_prompt)
            
        # If there's no prompt yet, provide a placeholder for app icon generation
        if not self.prompt_input.text():
            self.prompt_input.setText("clean professional app icon, [describe app purpose here], flat design, app icon, ios app icon")
            # Select the text for easy editing
            self.prompt_input.selectAll()
            self.prompt_input.setFocus()
        else:
            # If there is text, append app icon specific terms if not already present
            current_prompt = self.prompt_input.text()
            if "app icon" not in current_prompt.lower():
                enhanced_prompt = f"{current_prompt}, app icon, ios app icon"
                self.prompt_input.setText(enhanced_prompt)
                
        # Show a message to explain the preset
        QMessageBox.information(
            self,
            "App Icon Preset Applied",
            "App icon generation preset has been applied with the following optimizations:\n\n"
            f"• Resolution: {square_resolution}\n"
            f"• Steps: 25\n"
            f"• Guidance Scale: 7.0\n"
            f"• Batch Size: 4 (to provide options)\n"
            f"• Sampler: Optimized for detail\n"
            f"• Negative prompt: Optimized for clean icons\n\n"
            "For best results:\n"
            "• Describe your app purpose clearly in the prompt\n"
            "• Use terms like 'minimal', 'clean', 'flat design', or 'professional'\n"
            "• Generate multiple versions and pick the best one\n"
            "• For specific platforms add 'ios app icon' or 'android app icon'\n\n"
            "After generating, use 'App Icon Processing' from the Post Processing menu to add rounded corners and create multiple sizes."
        )
    
    def post_process_app_icon(self):
        """Apply post-processing to app icons including rounded corners and rescaling"""
        if not self.current_images or len(self.current_images) == 0 or self.current_image_index is None:
            QMessageBox.warning(
                self,
                "No Image Available",
                "Please generate an image first before using the post-processing options."
            )
            return
        
        # Create a dialog for post-processing options
        dialog = QDialog(self)
        dialog.setWindowTitle("Post-Process App Icon")
        dialog.setModal(True)
        
        # Set dialog size
        dialog.resize(450, 500)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create form layout for options
        form_layout = QFormLayout()
        
        # Rounded corners option
        corner_radius_label = QLabel("Corner Radius (%):")
        corner_radius_slider = QSpinBox()
        corner_radius_slider.setRange(0, 50)
        corner_radius_slider.setValue(20)
        corner_radius_slider.setSingleStep(5)
        corner_radius_slider.setToolTip("0% = Square, 50% = Full rounded corners")
        form_layout.addRow(corner_radius_label, corner_radius_slider)
        
        # Platform selection
        platform_label = QLabel("Target Platform:")
        platform_combo = QComboBox()
        platform_combo.addItems(["iOS", "Android", "Windows", "macOS", "All Platforms"])
        platform_combo.setCurrentText("iOS")
        form_layout.addRow(platform_label, platform_combo)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Directory:")
        output_dir_input = QLineEdit()
        output_dir_input.setText(os.path.join(self.outputs_dir, "app_icons"))
        output_dir_button = QPushButton("Browse...")
        
        # Connect browse button
        output_dir_button.clicked.connect(lambda: output_dir_input.setText(
            QFileDialog.getExistingDirectory(dialog, "Select Output Directory", output_dir_input.text())
        ))
        
        output_dir_layout.addWidget(output_dir_input)
        output_dir_layout.addWidget(output_dir_button)
        form_layout.addRow(output_dir_label, output_dir_layout)
        
        # Preview section
        preview_label = QLabel("Preview:")
        preview_image = QLabel()
        preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_image.setMinimumSize(200, 200)
        preview_image.setMaximumSize(200, 200)
        preview_image.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        
        # Create a function to update preview
        def update_preview():
            try:
                if not self.current_images or self.current_image_index is None:
                    return
                
                # Get the current image
                img = self.current_images[self.current_image_index].copy()
                
                # Apply corner radius
                radius_percent = corner_radius_slider.value()
                img = self.apply_corner_radius(img, radius_percent)
                
                # Convert to QPixmap for display
                img_data = img.convert("RGBA").tobytes("raw", "RGBA")
                qimage = QImage(img_data, img.width, img.height, img.width * 4, QImage.Format.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimage)
                
                # Scale pixmap to fit the preview label while maintaining aspect ratio
                pixmap = pixmap.scaled(preview_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                # Display the preview
                preview_image.setPixmap(pixmap)
            except Exception as e:
                print(f"Error updating preview: {str(e)}")
        
        # Connect slider to update preview
        corner_radius_slider.valueChanged.connect(update_preview)
        
        # Add process button
        process_button = QPushButton("Process Icon")
        
        # Connect process button
        def process_icon():
            try:
                if not self.current_images or self.current_image_index is None:
                    return
                
                # Get options
                radius_percent = corner_radius_slider.value()
                target_platform = platform_combo.currentText()
                output_directory = output_dir_input.text()
                
                # Create output directory if it doesn't exist
                os.makedirs(output_directory, exist_ok=True)
                
                # Get base icon name from the prompt or counter
                base_name = "_".join(self.prompt_input.text().split()[:3]).lower()
                base_name = ''.join(c for c in base_name if c.isalnum() or c == '_')
                if not base_name:
                    base_name = f"app_icon_{MainWindow.generation_counter:04d}"
                
                # Get the current image
                img = self.current_images[self.current_image_index].copy()
                
                # Apply corner radius
                img = self.apply_corner_radius(img, radius_percent)
                
                # Get resolutions based on platform
                resolutions = self.get_platform_resolutions(target_platform)
                
                # Process and save all resolutions
                processed_files = []
                for resolution_name, size in resolutions.items():
                    # Resize image
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    
                    # Create filename
                    filename = f"{base_name}_{resolution_name}.png"
                    filepath = os.path.join(output_directory, filename)
                    
                    # Save image
                    resized_img.save(filepath)
                    processed_files.append(filepath)
                
                # Show success message
                QMessageBox.information(
                    dialog,
                    "Processing Complete",
                    f"Successfully processed {len(processed_files)} icon sizes and saved to:\n{output_directory}"
                )
                
                # Open the folder in explorer
                if IS_WINDOWS:
                    os.startfile(output_directory)
                elif IS_MACOS:
                    import subprocess
                    subprocess.call(["open", output_directory])
                else:
                    import subprocess
                    subprocess.call(["xdg-open", output_directory])
                
                # Close dialog
                dialog.accept()
                
            except Exception as e:
                QMessageBox.critical(
                    dialog,
                    "Error",
                    f"An error occurred during processing: {str(e)}"
                )
        
        process_button.clicked.connect(process_icon)
        
        # Add cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        
        # Create button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(process_button)
        button_layout.addWidget(cancel_button)
        
        # Add layouts to main layout
        layout.addLayout(form_layout)
        layout.addWidget(preview_label)
        layout.addWidget(preview_image)
        layout.addLayout(button_layout)
        
        # Update the preview initially
        QTimer.singleShot(100, update_preview)
        
        # Show dialog
        dialog.exec()
    
    def apply_corner_radius(self, img, radius_percent):
        """Apply rounded corners to an image with the specified radius percentage"""
        # Make a copy to avoid modifying the original
        img = img.copy()
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Calculate radius in pixels (percentage of the smallest dimension)
        width, height = img.size
        min_dimension = min(width, height)
        radius = int(min_dimension * radius_percent / 100)
        
        # Don't do anything if radius is too small
        if radius < 1:
            return img
        
        # Create a transparent image with the same size
        rounded_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Create a mask with rounded corners
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Drawing rounded rectangle manually for compatibility with older Pillow versions
        # Draw the rectangles for the middle sections
        draw.rectangle([(radius, 0), (width - radius, height)], fill=255)  # Vertical rect
        draw.rectangle([(0, radius), (width, height - radius)], fill=255)  # Horizontal rect
        
        # Draw the four corner circles
        draw.pieslice([(0, 0), (radius * 2, radius * 2)], 180, 270, fill=255)  # Top-left
        draw.pieslice([(width - radius * 2, 0), (width, radius * 2)], 270, 360, fill=255)  # Top-right
        draw.pieslice([(0, height - radius * 2), (radius * 2, height)], 90, 180, fill=255)  # Bottom-left
        draw.pieslice([(width - radius * 2, height - radius * 2), (width, height)], 0, 90, fill=255)  # Bottom-right
        
        # Paste the original image using the mask
        rounded_img.paste(img, (0, 0), mask)
        
        return rounded_img
    
    def get_platform_resolutions(self, platform):
        """Get common icon resolutions for the specified platform"""
        resolutions = {}
        
        if platform == "iOS" or platform == "All Platforms":
            # iOS App Icon sizes
            resolutions.update({
                "ios_60x60": (60, 60),
                "ios_120x120": (120, 120), 
                "ios_180x180": (180, 180),
                "ios_1024x1024": (1024, 1024)  # App Store
            })
            
        if platform == "Android" or platform == "All Platforms":
            # Android App Icon sizes
            resolutions.update({
                "android_48x48": (48, 48),     # mdpi
                "android_72x72": (72, 72),     # hdpi
                "android_96x96": (96, 96),     # xhdpi
                "android_144x144": (144, 144), # xxhdpi
                "android_192x192": (192, 192), # xxxhdpi
                "android_512x512": (512, 512)  # Play Store
            })
            
        if platform == "Windows" or platform == "All Platforms":
            # Windows App Icon sizes
            resolutions.update({
                "windows_44x44": (44, 44),
                "windows_71x71": (71, 71),
                "windows_150x150": (150, 150),
                "windows_310x310": (310, 310)
            })
            
        if platform == "macOS" or platform == "All Platforms":
            # macOS App Icon sizes
            resolutions.update({
                "macos_16x16": (16, 16),
                "macos_32x32": (32, 32),
                "macos_64x64": (64, 64),
                "macos_128x128": (128, 128),
                "macos_256x256": (256, 256),
                "macos_512x512": (512, 512),
                "macos_1024x1024": (1024, 1024)
            })
            
        return resolutions
    
    def save_image(self):
        """Save the generated image(s)"""
        if not self.current_images:
            return
        
        # Inform the user that images are already auto-saved
        auto_save_info = QMessageBox()
        auto_save_info.setIcon(QMessageBox.Information)
        auto_save_info.setWindowTitle("Auto-Save Information")
        auto_save_info.setText("Images are automatically saved to the 'outputs' folder.")
        auto_save_info.setInformativeText("Do you still want to save to a custom location?")
        auto_save_info.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        choice = auto_save_info.exec()
        
        if choice == QMessageBox.No:
            return
        
        if len(self.current_images) == 1:
            # Single image save
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
            )
            
            if file_path:
                self.current_images[0].save(file_path)
                self.status_label.setText(f"Saved image to {file_path}")
        else:
            # Batch save - ask user what they want to do
            options = QMessageBox.question(
                self,
                "Save Images",
                "Do you want to save all images or just the currently displayed one?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.SaveAll | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )
            
            if options == QMessageBox.StandardButton.Cancel:
                return
                
            # Determine which images to save
            images_to_save = [self.current_images[self.current_image_index]] if options == QMessageBox.StandardButton.Save else self.current_images
            
            # Get directory to save to
            if len(images_to_save) > 1:
                save_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory to Save Images"
                )
                
                if save_dir:
                    # Generate base filename from prompt
                    base_name = "_".join(self.prompt_input.text().split()[:3]).lower()
                    base_name = "".join(c for c in base_name if c.isalnum() or c == '_')
                    
                    # If base_name is empty (e.g., prompt had no alphanumeric chars), use "image" instead
                    if not base_name:
                        base_name = "image"
                    
                    # Save each image with the same format as auto-save: counter_promptwords_seed.png
                    for i, image in enumerate(images_to_save):
                        # Increment counter for each custom saved image too
                        MainWindow.generation_counter += 1
                        # Get the seed if available
                        seed = self.generation_thread.generated_seeds[i] if hasattr(self.generation_thread, 'generated_seeds') and i < len(self.generation_thread.generated_seeds) else "unknown"
                        # Use the same format as auto_save_image
                        file_path = os.path.join(save_dir, f"{MainWindow.generation_counter:04d}_{base_name}_seed{seed}.png")
                        image.save(file_path)
                    
                    self.status_label.setText(f"Saved {len(images_to_save)} images to {save_dir}")
            else:
                # Save single image
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Image",
                    "",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
                )
                
                if file_path:
                    images_to_save[0].save(file_path)
                    self.status_label.setText(f"Saved image to {file_path}")

    def auto_save_image(self, image, index, seed):
        """Automatically save the generated image to the outputs folder"""
        try:
            # Increment the generation counter for this new image
            MainWindow.generation_counter += 1
            
            # Generate base filename from prompt (first 3 words)
            base_name = "_".join(self.prompt_input.text().split()[:3]).lower()
            base_name = "".join(c for c in base_name if c.isalnum() or c == '_')
            
            # If base_name is empty (e.g., prompt had no alphanumeric chars), use "image" instead
            if not base_name:
                base_name = "image"
            
            # Create filename with counter first, then prompt and seed
            # Format: 0001_promptwords_seed123456.png
            file_name = f"{MainWindow.generation_counter:04d}_{base_name}_seed{seed}.png"
            file_path = os.path.join(self.outputs_dir, file_name)
            
            # Save the image
            image.save(file_path)
            print(f"Auto-saved image to {file_path}")
            
            # Update status label to inform about auto-saving
            try:
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Generated image {index+1}/{len(self.current_images)} (seed: {seed}) - Auto-saved to outputs folder")
            except RuntimeError:
                # Status label was deleted
                print(f"Status label deleted, can't update. Auto-saved image to {file_path}")
        except Exception as e:
            # Log other unexpected errors but still try to save the image
            print(f"Unexpected error in auto_save_image: {str(e)}")
            try:
                # Try to save the image even if there was an error updating the UI
                image.save(file_path)
                print(f"Auto-saved image despite errors: {file_path}")
            except:
                print("Failed to save image due to error")

def scan_local_models():
    """Scan the models directory for safetensors and ckpt files"""
    model_files = []
    for ext in [".safetensors", ".ckpt"]:
        pattern = os.path.join(LOCAL_MODELS_DIR, f"*{ext}")
        found_files = glob.glob(pattern)
        model_files.extend(found_files)
    return model_files

def auto_import_local_models():
    """
    Automatically import models from the models directory
    with special handling for Pony models to be imported as SDXL
    """
    model_files = scan_local_models()
    
    # Skip if no model files found
    if not model_files:
        return {"count": 0, "message": ""}
    
    imported_count = 0
    pony_count = 0
    imported_models = []
    
    for model_path in model_files:
        # Skip already imported models
        if any(model.file_path == model_path for model in LOCAL_MODELS.values()):
            continue
        
        # Extract default name from the file path
        default_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Determine model type - check for Pony models
        filename_lower = default_name.lower()
        is_pony_model = False
        
        if ("pony" in filename_lower or 
            "ponyrealism" in filename_lower or 
            "ponydiffusion" in filename_lower or
            "pony_diffusion" in filename_lower or
            "pony-diffusion" in filename_lower or
            "mlp" in filename_lower):  # My Little Pony abbreviation
            model_type = "Stable Diffusion XL"  # Set all Pony models as SDXL
            is_pony_model = True
            pony_count += 1
        else:
            # Default to SD 1.5 for others
            model_type = "Stable Diffusion 1.5"
        
        # Create description with special note for Pony models
        description = f"Auto-imported local model: {default_name}"
        if is_pony_model:
            description += " (Pony model - configured for SDXL architecture)"
        else:
            description += f" ({model_type})"
        
        # Create model info and add to LOCAL_MODELS
        model_info = LocalModelInfo(
            name=default_name,
            file_path=model_path,
            model_type=model_type,
            description=description
        )
        
        # For Pony models, set the custom negative prompt manually
        if is_pony_model:
            # Get the base config first
            config = model_info.get_config()
            # Then manually override the negative prompt
            config["negative_prompt"] = DEFAULT_NEGATIVE_PROMPTS["pony"]
            # Store the modified config back in the model info
            model_info.pony_override = True  # Flag to indicate this is a Pony model
            # Note: The next time get_config is called, we'll need to restore this override
        
        LOCAL_MODELS[default_name] = model_info
        imported_count += 1
        imported_models.append(default_name)
    
    # Show a message if models were imported
    if imported_count > 0:
        message = f"Auto-imported {imported_count} local models:\n\n"
        message += "\n".join(imported_models)
        
        if pony_count > 0:
            message += f"\n\n{pony_count} Pony models were automatically configured as SDXL."
        
        # Note: This needs to be shown after the QApplication is created, which is handled in the main block
        imported_info = {
            "count": imported_count, 
            "message": message
        }
        return imported_info
    
    return {"count": 0, "message": ""}

class AddLocalModelDialog(QDialog):
    """Dialog for adding a local model"""
    def __init__(self, parent=None, model_path=None):
        super().__init__(parent)
        self.setWindowTitle("Add Local Model")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Model name field
        self.name_input = QLineEdit()
        if model_path:
            # Extract a default name from the file path
            default_name = os.path.splitext(os.path.basename(model_path))[0]
            self.name_input.setText(default_name)
        form.addRow("Model Name:", self.name_input)
        
        # Model type selection
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(MODEL_TYPES.keys())
        form.addRow("Model Type:", self.model_type_combo)
        
        # Model path field
        self.path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        if model_path:
            self.path_input.setText(model_path)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_model)
        self.path_layout.addWidget(self.path_input)
        self.path_layout.addWidget(self.browse_button)
        form.addRow("Model File:", self.path_layout)
        
        # Description field
        self.description_input = QLineEdit()
        form.addRow("Description:", self.description_input)
        
        layout.addLayout(form)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.add_button = QPushButton("Add Model")
        self.add_button.clicked.connect(self.accept)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.add_button)
        
        layout.addLayout(button_layout)
    
    def browse_model(self):
        """Open file dialog to select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model File", 
            LOCAL_MODELS_DIR, 
            "Model Files (*.safetensors *.ckpt)"
        )
        
        if file_path:
            self.path_input.setText(file_path)
            # Extract a default name if none set
            if not self.name_input.text():
                default_name = os.path.splitext(os.path.basename(file_path))[0]
                self.name_input.setText(default_name)
    
    def get_model_info(self):
        """Get the model info from dialog fields"""
        if not self.path_input.text() or not self.name_input.text():
            return None
            
        return LocalModelInfo(
            name=self.name_input.text(),
            file_path=self.path_input.text(),
            model_type=self.model_type_combo.currentText(),
            description=self.description_input.text()
        )

class LocalModelsDialog(QDialog):
    """Dialog for managing local models"""
    models_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Local Models")
        self.setMinimumSize(700, 400)
        
        layout = QVBoxLayout(self)
        
        # Models list
        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.models_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Model")
        self.add_button.clicked.connect(self.add_model)
        
        self.import_button = QPushButton("Import from Models Folder")
        self.import_button.clicked.connect(self.import_models)
        
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_model)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Load models
        self.refresh_models_list()
    
    def refresh_models_list(self):
        """Refresh the list of models"""
        self.models_list.clear()
        
        for name, model_info in LOCAL_MODELS.items():
            item = QListWidgetItem(f"{name} ({model_info.model_type})")
            item.setToolTip(f"Path: {model_info.file_path}\nDescription: {model_info.description}")
            self.models_list.addItem(item)
    
    def add_model(self):
        """Open dialog to add a new model"""
        dialog = AddLocalModelDialog(self)
        if dialog.exec():
            model_info = dialog.get_model_info()
            if model_info:
                LOCAL_MODELS[model_info.name] = model_info
                self.refresh_models_list()
                self.models_updated.emit()
    
    def import_models(self):
        """Import models from the models directory"""
        model_files = scan_local_models()
        
        if not model_files:
            QMessageBox.information(
                self,
                "No Models Found",
                f"No model files found in {LOCAL_MODELS_DIR}.\n\n"
                "Please place your .safetensors or .ckpt files in this folder."
            )
            return
            
        for model_path in model_files:
            # Skip already imported models
            if any(model.file_path == model_path for model in LOCAL_MODELS.values()):
                continue
                
            dialog = AddLocalModelDialog(self, model_path)
            if dialog.exec():
                model_info = dialog.get_model_info()
                if model_info:
                    LOCAL_MODELS[model_info.name] = model_info
        
        self.refresh_models_list()
        self.models_updated.emit()
    
    def remove_model(self):
        """Remove selected model from registry"""
        selected_items = self.models_list.selectedItems()
        if not selected_items:
            return
            
        item = selected_items[0]
        model_name = item.text().split(" (")[0]
        
        if model_name in LOCAL_MODELS:
            confirmation = QMessageBox.question(
                self,
                "Confirm Removal",
                f"Remove {model_name} from the registry?\n\n"
                "This will only remove the model from the list, not delete the file.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if confirmation == QMessageBox.StandardButton.Yes:
                del LOCAL_MODELS[model_name]
                self.refresh_models_list()
                self.models_updated.emit()

# Custom MacOS Combo Box that fixes selection issues
class MacComboBox(QComboBox):
    """Custom combo box for macOS that ensures dropdowns work properly"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use editable combo box with read-only behavior to fix macOS dropdown issues
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #AAA;
                border-radius: 4px;
                background-color: #333;
                color: white;
            }
            QComboBox::drop-down {
                width: 20px;
                background-color: #444;
                border-left: 1px solid #888;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                border: 1px solid #AAA;
                selection-background-color: #666;
                selection-color: white;
                color: white;
            }
            QComboBox QLineEdit {
                background-color: #333;
                color: white;
                border: none;
                selection-background-color: #666;
            }
        """)
        
    def showPopup(self):
        """Ensure popup displays correctly on macOS"""
        super().showPopup()
        # Force alternating row colors for better visibility
        self.view().setAlternatingRowColors(True)

class MacDropdownDialog(QDialog):
    """A dialog-based dropdown for macOS that's more reliable than QMenu or QComboBox"""
    
    def __init__(self, items, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.selected_item = None
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create list widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #333;
                color: white;
                border: 2px solid #888;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 8px 20px;
            }
            QListWidget::item:selected, QListWidget::item:hover {
                background-color: #666;
            }
        """)
        
        # Add items
        for item in items:
            self.list_widget.addItem(item)
        
        # Connect signal
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        
        # Add to layout
        layout.addWidget(self.list_widget)
        
        # Set size
        self.setMinimumWidth(300)
        self.setMaximumHeight(400)
        
    def on_item_clicked(self, item):
        self.selected_item = item.text()
        self.accept()

class MacDropdownButton(QPushButton):
    """Button-based dropdown for macOS that uses a dialog for selection"""
    
    # Define a proper PyQt signal
    currentTextChanged = pyqtSignal(str)
    
    def __init__(self, items=None, parent=None):
        super().__init__(parent)
        
        # Make sure the button has adequate size and visibility
        self.setMinimumHeight(35)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Add arrow indicator with high visibility text
        self.setText("▼ Click to Select")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        # Highly visible styling with brighter colors
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px 25px 8px 10px;
                border: 2px solid #888;
                border-radius: 4px;
                background-color: #444;
                color: white;
                min-height: 35px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #555;
                border-color: #aaa;
            }
            QPushButton:pressed {
                background-color: #666;
            }
        """)
        
        self._items = []
        self._current_text = ""
        self._current_index = -1
        
        # Connect our own click handler
        self.clicked.connect(self._show_dropdown)
        
        # Initialize with items if provided
        if items:
            self.addItems(items)
    
    def _show_dropdown(self):
        """Show a dialog-based dropdown when button is clicked"""
        if not self._items:
            return
            
        # Create the dialog
        dialog = MacDropdownDialog(self._items, self)
        
        # Position the dialog under the button
        pos = self.mapToGlobal(self.rect().bottomLeft())
        dialog.move(pos)
        
        # Show the dialog modally
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_item:
            index = self._items.index(dialog.selected_item)
            self._on_item_selected(index, dialog.selected_item)
    
    def addItems(self, items):
        """Add multiple items to the dropdown"""
        self._items.extend(items)
        
        # Set initial text if we have items and no current selection
        if items and not self._current_text:
            self.setCurrentIndex(0)
    
    def addItem(self, item):
        """Add a single item to the dropdown"""
        self.addItems([item])
    
    def _on_item_selected(self, index, text):
        """Handle item selection"""
        self._current_index = index
        self._current_text = text
        
        # Update button text - keep the arrow but add the selection
        self.setText(f"{text} ▼")
        
        # Emit the signal
        self.currentTextChanged.emit(text)
    
    def currentText(self):
        """Get the current selected text"""
        return self._current_text
    
    def currentIndex(self):
        """Get the current selected index"""
        return self._current_index
    
    def setCurrentIndex(self, index):
        """Set the current selection by index"""
        if 0 <= index < len(self._items):
            self._current_index = index
            self._current_text = self._items[index]
            self.setText(f"{self._current_text} ▼")
    
    def setCurrentText(self, text):
        """Set the current selection by text"""
        if text in self._items:
            index = self._items.index(text)
            self.setCurrentIndex(index)
    
    def clear(self):
        """Clear all items"""
        self._items = []
        self._current_text = ""
        self._current_index = -1
        self.setText("▼ Click to Select")
    
    def count(self):
        """Get the number of items"""
        return len(self._items)
    
    def findText(self, text):
        """Find the index of the given text"""
        try:
            return self._items.index(text)
        except ValueError:
            return -1
    
    # Remove the old currentTextChanged method that registered handlers

# Create a centralized error handling and logging system
class ErrorHandler:
    """Centralized error handling and logging system"""
    
    LOG_LEVEL = logging.INFO  # Default log level
    LOG_FILE = "dream_pixel_forge.log"
    
    @classmethod
    def setup_logging(cls):
        """Set up logging configuration"""
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        # Create a logger
        cls.logger = logging.getLogger("DreamPixelForge")
        cls.logger.info("Logging initialized")
        
    @classmethod
    def log_info(cls, message):
        """Log informational message"""
        if not hasattr(cls, 'logger'):
            cls.setup_logging()
        cls.logger.info(message)
        
    @classmethod
    def log_warning(cls, message):
        """Log warning message"""
        if not hasattr(cls, 'logger'):
            cls.setup_logging()
        cls.logger.warning(message)
        
    @classmethod
    def log_error(cls, message, exc_info=None):
        """Log error message with optional exception info"""
        if not hasattr(cls, 'logger'):
            cls.setup_logging()
        if exc_info:
            cls.logger.error(f"{message}\n{traceback.format_exc()}")
        else:
            cls.logger.error(message)
    
    @classmethod
    def handle_ui_error(cls, parent, error, title="Error", details=None, show_dialog=True):
        """Handle UI errors with appropriate user feedback"""
        error_msg = str(error)
        
        # Log the error
        cls.log_error(f"UI Error: {error_msg}", exc_info=error)
        
        # Only show dialog if requested and if parent widget is still valid
        # In PyQt6, we should check if the widget exists and is visible
        if show_dialog and parent and parent.isVisible():
            try:
                msg_box = QMessageBox(parent)
                msg_box.setIcon(QMessageBox.Icon.Critical)
                msg_box.setWindowTitle(title)
                msg_box.setText(error_msg)
                
                if details:
                    msg_box.setDetailedText(details)
                    
                msg_box.exec()
            except Exception as dialog_error:
                # If we can't show a dialog, fall back to console output
                cls.log_error(f"Failed to show error dialog: {str(dialog_error)}")
                print(f"ERROR: {title} - {error_msg}")
                
        return error  # Return the error to allow for chaining

    @classmethod
    def handle_generation_error(cls, parent, error, title="Generation Error"):
        """Specific handler for image generation errors"""
        error_msg = str(error)
        traceback_info = traceback.format_exc()
        
        # Log the error
        cls.log_error(f"Generation Error: {error_msg}", exc_info=error)
        
        # Provide more user-friendly error messages for common cases
        if "CUDA out of memory" in error_msg:
            friendly_msg = "Your GPU ran out of memory. Try using a smaller resolution or reducing batch size."
            details = f"Original error: {error_msg}\n\n{traceback_info}"
            return cls.handle_ui_error(parent, friendly_msg, title, details)
        elif "TypeError: argument of type 'NoneType'" in error_msg:
            friendly_msg = "Model configuration error. This model may not be fully compatible."
            details = f"Original error: {error_msg}\n\n{traceback_info}"
            return cls.handle_ui_error(parent, friendly_msg, title, details)
        else:
            # For unknown errors, show the full traceback in details
            return cls.handle_ui_error(parent, error_msg, title, traceback_info)
    
    @classmethod
    def safe_ui_operation(cls, parent, operation_func, error_title="UI Operation Error"):
        """Safely execute a UI operation with proper error handling"""
        try:
            return operation_func()
        except Exception as e:
            cls.handle_ui_error(parent, e, error_title)
            return None

# Set up logging when module is imported
ErrorHandler.setup_logging()

class CollapsibleSection(QWidget):
    """A custom collapsible section widget with disclosure triangle for macOS style"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Header layout with toggle button
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Toggle button with arrow
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("disclosureTriangle")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.on_toggle)
        self.toggle_button.setText("▶")  # Right-pointing triangle when collapsed
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        
        # Add to header layout
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        
        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(25, 5, 5, 5)  # Indent content
        
        # Add widgets to main layout
        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(self.content_widget)
        
        # Set initial state (collapsed)
        self.content_widget.setVisible(False)
        
        # Style for macOS
        self.setStyleSheet("""
            #collapsibleSection {
                border: 1px solid #666;
                border-radius: 3px;
                background-color: #333;
                margin: 2px;
            }
            #disclosureTriangle {
                background-color: transparent;
                border: none;
                color: #ccc;
                font-size: 12px;
                font-weight: bold;
            }
            #disclosureTriangle:hover {
                color: white;
            }
        """)
    
    def on_toggle(self, checked):
        """Toggle visibility of content"""
        self.content_widget.setVisible(checked)
        self.toggle_button.setText("▼" if checked else "▶")  # Down arrow when expanded, right arrow when collapsed
    
    def addWidget(self, widget):
        """Add a widget to the content layout"""
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        """Add a layout to the content layout"""
        self.content_layout.addLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Auto-import local models on startup
    import_result = auto_import_local_models()
    
    window = MainWindow()
    window.show()
    
    # Show message about auto-imported models if any
    if import_result and import_result["count"] > 0:
        QMessageBox.information(
            window,
            "Models Auto-Imported",
            import_result["message"]
        )
    
    sys.exit(app.exec()) 