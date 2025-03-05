import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
                            QComboBox, QMessageBox, QGroupBox, QRadioButton, 
                            QTabWidget, QListWidget, QListWidgetItem, QDialog,
                            QFormLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QAction
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, KandinskyV22Pipeline, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, PNDMScheduler, DDPMScheduler, DEISMultistepScheduler, DPMSolverSDEScheduler, KarrasVeScheduler
import torch
from PIL import Image
import io
import traceback
from huggingface_hub import scan_cache_dir, HfFolder, model_info
import time
import requests
import json
import random
import glob
import platform

# Local models directory
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# Detect macOS for MPS (Metal) support
IS_MACOS = platform.system() == 'Darwin'
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
                torch.cuda.empty_cache()
                print("CUDA is available, using GPU")
                print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            elif device == "mps":
                print("Using Apple Metal (MPS) for acceleration")
                # No equivalent cache clearing for MPS
            else:
                print("Neither CUDA nor MPS is available, using CPU - generation will be slow")
            
            # Check if model is already downloaded
            if not is_model_downloaded(model_id):
                print(f"Model {model_id} not found in cache, will be downloaded")
                self.progress.emit(0, f"First time using {model_id}. Downloading (~{size_gb:.1f}GB)...")
                
                # Start a thread to provide ongoing download status
                self.download_tracker = DownloadTracker(model_id, size_gb)
                self.download_tracker.progress.connect(lambda msg: self.progress.emit(0, msg))
                self.download_tracker.start()
                
                # Give the tracker a moment to show its first message
                time.sleep(0.5)
                QApplication.processEvents()
            
            # Initialize pipeline
            print(f"Initializing pipeline for model: {model_id}")
            
            # Check if this is a local model
            is_local = self.model_config.get("is_local", False)
            
            # Get appropriate torch dtype
            torch_dtype = get_torch_dtype()
            
            if is_local and os.path.exists(model_id) and model_id.endswith((".safetensors", ".ckpt")):
                print(f"Loading local model from: {model_id}")
                self.progress.emit(0, f"Loading local model from: {os.path.basename(model_id)}...")
                
                try:
                    # Load from single file (used for Civitai models)
                    self.pipe = pipeline_class.from_single_file(
                        model_id,
                        torch_dtype=torch_dtype
                    )
                    print("Local model loaded successfully")
                except Exception as e:
                    error_msg = f"Error loading local model: {str(e)}"
                    print(error_msg)
                    # Try the regular from_pretrained as a fallback
                    print("Trying alternative loading method...")
                    self.pipe = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        local_files_only=True
                    )
                
                # Fix UNet for local SDXL models that expect additional embeddings
                try:
                    if pipeline_class == StableDiffusionXLPipeline or "XL" in model_id:
                        # Check if UNet has addition_embed_type set
                        if hasattr(self.pipe.unet, "config") and hasattr(self.pipe.unet.config, "addition_embed_type"):
                            if self.pipe.unet.config.addition_embed_type is not None:
                                print("Detected SDXL model that requires additional embeddings")
                                print("Setting addition_embed_type to None to prevent errors")
                                self.pipe.unet.config.addition_embed_type = None
                                print("UNet configuration updated for compatibility with local model")
                except Exception as e:
                    print(f"Warning: Could not fix UNet configuration: {str(e)}")
                    print("Generation might fail with this model")
            else:
                # Load remote model from Hugging Face Hub
                self.pipe = pipeline_class.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype
                )
            
            # Special handling for Pony Diffusion - Set clip_skip=2
            if "pony" in model_id.lower() and hasattr(self.pipe, "text_encoder"):
                print("Setting clip_skip=2 for Pony Diffusion model")
                # For SDXL models with text_encoder and text_encoder_2
                if hasattr(self.pipe, "text_encoder_2"):
                    # Store only the penultimate layer's hidden states in both encoders
                    self.pipe.text_encoder.config.clip_skip = 2
                    self.pipe.text_encoder_2.config.clip_skip = 2
                else:
                    # For standard SD models with a single text encoder
                    self.pipe.text_encoder.config.clip_skip = 2
                print("Clip skip set to 2 for better results with Pony Diffusion")
            
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
                print(f"Moving model to {device.upper()}...")
                self.pipe = self.pipe.to(device)
                
                # Try to enable memory efficient attention if CUDA is available
                if device == "cuda":
                    try:
                        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                            print("Enabling memory efficient attention...")
                            self.pipe.enable_xformers_memory_efficient_attention()
                    except Exception as e:
                        print(f"Could not enable memory efficient attention: {str(e)}")
                        print("Continuing without memory optimization...")

            # Generate the images in batch
            self.progress.emit(0, f"Starting generation of {self.batch_size} images...")
            print(f"Starting generation with prompt: {self.prompt}")
            print(f"Negative prompt: {self.negative_prompt}")
            print(f"Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            print(f"Resolution: {self.width}x{self.height}, Batch size: {self.batch_size}")
            
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
                    print(f"Using seed for image {i+1}: {current_seed}")
                else:
                    # For subsequent images or if no seed was provided, use random seeds
                    generator = torch.Generator(device=get_device() if get_device() != "mps" else "cpu")
                    # Generate a seed that fits in a 32-bit integer to avoid overflow
                    current_seed = random.randint(0, 2147483647)
                    generator.manual_seed(current_seed)
                    print(f"Using random seed for image {i+1}: {current_seed}")
                
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
                            print(f"Using sampler: {self.sampler}")
                            # Get the config from the current scheduler
                            old_config = self.pipe.scheduler.config
                            # Create new scheduler with the same config
                            new_scheduler = schedulers[self.sampler].from_config(old_config)
                            # Replace the scheduler
                            self.pipe.scheduler = new_scheduler
                        else:
                            print(f"Unknown sampler: {self.sampler}, using default")
                    except Exception as e:
                        print(f"Error setting sampler: {str(e)}")
                        print("Using default sampler")
                
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
        self.setWindowTitle("DreamPixelForge - Text to Image")
        self.setMinimumSize(800, 600)
        
        # Create menu bar
        self.create_menu()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient()
        self.ollama_available = self.ollama_client.is_available()
        self.ollama_models = []
        if self.ollama_available:
            self.ollama_models = self.ollama_client.list_models()
        self.ollama_thread = None
        
        # Create input section
        input_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        
        # Create a tabbed interface for model selection
        self.model_tabs = QTabWidget()
        
        # Hugging Face models tab
        hf_tab = QWidget()
        hf_layout = QVBoxLayout(hf_tab)
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        hf_layout.addWidget(self.model_combo)
        self.model_tabs.addTab(hf_tab, "Hugging Face Models")
        
        # Local models tab
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        self.local_model_combo = QComboBox()
        self.local_model_combo.currentTextChanged.connect(self.on_local_model_changed)
        
        local_buttons_layout = QHBoxLayout()
        manage_models_btn = QPushButton("Manage Models")
        manage_models_btn.clicked.connect(self.open_manage_models)
        local_buttons_layout.addWidget(self.local_model_combo)
        local_buttons_layout.addWidget(manage_models_btn)
        
        local_layout.addLayout(local_buttons_layout)
        self.model_tabs.addTab(local_tab, "Local Models")
        
        # Connect tab change to update model info
        self.model_tabs.currentChanged.connect(self.update_model_info_from_tabs)
        
        # Add a model info label
        self.model_info = QLabel(AVAILABLE_MODELS["Stable Diffusion 1.5"]["description"])
        self.model_info.setWordWrap(True)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_tabs)
        input_layout.addLayout(model_layout)
        input_layout.addWidget(self.model_info)
        
        # Prompt input
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("Prompt:")
        self.prompt_input = QLineEdit()
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_input)
        input_layout.addLayout(prompt_layout)
        
        # Negative prompt input
        neg_prompt_layout = QHBoxLayout()
        neg_prompt_label = QLabel("Negative Prompt:")
        self.neg_prompt_input = QLineEdit()
        self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPTS["sd15"])
        neg_prompt_layout.addWidget(neg_prompt_label)
        neg_prompt_layout.addWidget(self.neg_prompt_input)
        input_layout.addLayout(neg_prompt_layout)
        
        # Ollama prompt enhancement section
        if self.ollama_available and self.ollama_models:
            ollama_group = QGroupBox("Ollama Prompt Enhancement")
            ollama_layout = QVBoxLayout()
            
            # Model selection
            ollama_model_layout = QHBoxLayout()
            ollama_model_label = QLabel("Ollama Model:")
            self.ollama_model_combo = QComboBox()
            self.ollama_model_combo.addItems(self.ollama_models)
            refresh_ollama_button = QPushButton("Refresh")
            refresh_ollama_button.clicked.connect(self.refresh_ollama_models)
            ollama_model_layout.addWidget(ollama_model_label)
            ollama_model_layout.addWidget(self.ollama_model_combo)
            ollama_model_layout.addWidget(refresh_ollama_button)
            ollama_layout.addLayout(ollama_model_layout)
            
            # Input mode selection
            input_mode_layout = QHBoxLayout()
            self.description_radio = QRadioButton("Description to Tags")
            self.tags_radio = QRadioButton("Enhance Tags")
            self.tags_radio.setChecked(True)  # Default to tag enhancement
            input_mode_layout.addWidget(self.description_radio)
            input_mode_layout.addWidget(self.tags_radio)
            ollama_layout.addLayout(input_mode_layout)
            
            # Enhance button and input for enhancement
            enhance_layout = QHBoxLayout()
            self.enhance_input = QLineEdit()
            self.enhance_input.setPlaceholderText("Enter prompt to enhance")
            self.enhance_button = QPushButton("Enhance Prompt")
            self.enhance_button.clicked.connect(self.enhance_prompt)
            enhance_layout.addWidget(self.enhance_input)
            enhance_layout.addWidget(self.enhance_button)
            ollama_layout.addLayout(enhance_layout)
            
            ollama_group.setLayout(ollama_layout)
            input_layout.addWidget(ollama_group)
        else:
            # Show a message when Ollama is not available
            ollama_group = QGroupBox("Ollama Prompt Enhancement")
            ollama_layout = QVBoxLayout()
            
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
            
            ollama_group.setLayout(ollama_layout)
            input_layout.addWidget(ollama_group)
        
        # Parameters section
        params_layout = QHBoxLayout()
        
        # Resolution preset
        resolution_layout = QVBoxLayout()
        resolution_label = QLabel("Resolution:")
        self.resolution_combo = QComboBox()
        
        # Initialize with default model's resolutions
        default_model = "Stable Diffusion 1.5"
        resolution_presets = AVAILABLE_MODELS[default_model]["resolution_presets"]
        self.resolution_combo.addItems(resolution_presets.keys())
        
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(self.resolution_combo)
        params_layout.addLayout(resolution_layout)
        
        # Number of steps
        steps_layout = QVBoxLayout()
        steps_label = QLabel("Number of Steps:")
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 100)
        self.steps_input.setValue(30)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_input)
        params_layout.addLayout(steps_layout)
        
        # Guidance scale
        guidance_layout = QVBoxLayout()
        guidance_label = QLabel("Guidance Scale:")
        self.guidance_input = QDoubleSpinBox()
        self.guidance_input.setRange(1.0, 20.0)
        self.guidance_input.setValue(7.5)
        self.guidance_input.setSingleStep(0.5)
        guidance_layout.addWidget(guidance_label)
        guidance_layout.addWidget(self.guidance_input)
        params_layout.addLayout(guidance_layout)
        
        # Seed input
        seed_layout = QVBoxLayout()
        seed_label = QLabel("Seed:")
        seed_control_layout = QHBoxLayout()
        self.seed_input = QSpinBox()
        self.seed_input.setRange(-1, 2147483647)  # Max int32 value
        self.seed_input.setValue(-1)  # -1 means random seed
        self.seed_input.setToolTip("Use -1 for random seed")
        self.random_seed_button = QPushButton("🎲")  # Dice emoji for random
        self.random_seed_button.setToolTip("Generate random seed")
        self.random_seed_button.setMaximumWidth(30)
        self.random_seed_button.clicked.connect(self.generate_random_seed)
        seed_control_layout.addWidget(self.seed_input)
        seed_control_layout.addWidget(self.random_seed_button)
        seed_layout.addWidget(seed_label)
        seed_layout.addLayout(seed_control_layout)
        params_layout.addLayout(seed_layout)
        
        # Sampler selection
        sampler_layout = QVBoxLayout()
        sampler_label = QLabel("Sampler:")
        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(SAMPLERS.keys())
        sampler_layout.addWidget(sampler_label)
        sampler_layout.addWidget(self.sampler_combo)
        params_layout.addLayout(sampler_layout)
        
        # Batch size
        batch_layout = QVBoxLayout()
        batch_label = QLabel("Batch Size:")
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 10)  # Allow generating up to 10 images at once
        self.batch_input.setValue(1)  # Default to single image
        self.batch_input.setToolTip("Number of images to generate in one batch")
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_input)
        params_layout.addLayout(batch_layout)
        
        input_layout.addLayout(params_layout)
        layout.addLayout(input_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Images")
        self.generate_button.clicked.connect(self.generate_image)
        layout.addWidget(self.generate_button)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        layout.addWidget(self.image_label)
        
        # Create save button
        self.save_button = QPushButton("Save Image to Custom Location")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
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
        self.seed_input.setValue(random_seed)
        
    def generate_image(self):
        if not self.prompt_input.text():
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
        seed_value = self.seed_input.value()
        if seed_value == -1:
            seed_value = None
            self.status_label.setText("Generating with random seed...")
        else:
            self.status_label.setText(f"Generating with seed: {seed_value}...")
            
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
            # Get the layout containing the image navigation
            layout = self.centralWidget().layout()
            
            # Remove the old navigation widgets
            while self.image_nav_layout.count():
                item = self.image_nav_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Remove the layout itself
            layout.removeItem(self.image_nav_layout)
            
            # Delete the navigation layout attribute
            delattr(self, 'image_nav_layout')
            
        # Create and start the generation thread
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

    def handle_progress(self, progress, message):
        """Handle progress updates from generation thread"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        
    def handle_error(self, error_message):
        """Handle errors from generation thread"""
        self.status_label.setText("Error: Generation failed")
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show error message in dialog
        QMessageBox.critical(self, "Generation Error", error_message)
        
    def handle_generation_finished(self, images):
        """Handle completion of generation thread"""
        self.generate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation complete! All images auto-saved to outputs folder.")
        
    def handle_image_ready(self, image, index, total):
        """Handle each image as it completes generation"""
        try:
            # Store the image in the current_images list
            if not hasattr(self, 'current_images') or self.current_images is None or len(self.current_images) != total:
                self.current_images = [None] * total

            # Add the image to the list at the correct index
            self.current_images[index] = image
            
            # Update status label with progress
            seed = self.generation_thread.generated_seeds[index]
            try:
                self.status_label.setText(f"Generated image {index+1}/{total} (seed: {seed})")
            except RuntimeError:
                # Status label was deleted
                print(f"Status label was deleted, can't update. Generated image {index+1}/{total} (seed: {seed})")
            
            # Auto-save image to outputs folder
            self.auto_save_image(image, index, seed)
            
            # Initialize UI for image browsing if needed
            if total > 1 and not hasattr(self, 'image_nav_layout'):
                # Create navigation buttons for browsing images
                self.image_nav_layout = QHBoxLayout()
                
                self.prev_button = QPushButton("Previous")
                self.prev_button.clicked.connect(self.show_previous_image)
                self.prev_button.setEnabled(False)  # Disabled at first image
                
                self.image_counter_label = QLabel(f"Image {index+1}/{total}")
                self.image_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.next_button = QPushButton("Next")
                self.next_button.clicked.connect(self.show_next_image)
                self.next_button.setEnabled(index < total - 1)  # Only enabled if there are more images
                
                self.image_nav_layout.addWidget(self.prev_button)
                self.image_nav_layout.addWidget(self.image_counter_label)
                self.image_nav_layout.addWidget(self.next_button)
                
                # Find the layout that contains the image label
                layout = self.centralWidget().layout()
                layout.insertLayout(layout.count() - 1, self.image_nav_layout)  # Insert before the save button
            elif hasattr(self, 'image_counter_label'):
                # Update the counter if it already exists
                try:
                    self.image_counter_label.setText(f"Image {index+1}/{total}")
                    self.prev_button.setEnabled(index > 0)
                    self.next_button.setEnabled(index < total - 1)
                except RuntimeError:
                    # UI element was likely deleted during model change
                    print("UI update error: Navigation widgets have been deleted")
            
            # Set the current image index
            self.current_image_index = index
            
            # Display the image
            self.display_image(index)
            
            # Enable the save button after the first image
            try:
                self.save_button.setEnabled(True)
                
                # Update save button text based on total images
                if total > 1:
                    self.save_button.setText("Save Images to Custom Location")
                else:
                    self.save_button.setText("Save Image to Custom Location")
            except RuntimeError:
                # UI element was likely deleted during model change
                print("UI update error: Save button has been deleted")
        except RuntimeError as e:
            # Handle Qt object deleted errors
            print(f"UI update error (normal if model was changed): {str(e)}")
        except Exception as e:
            # Log other unexpected errors
            print(f"Unexpected error in handle_image_ready: {str(e)}")

    def display_image(self, index):
        """Display the image at the specified index"""
        try:
            if not self.current_images or index >= len(self.current_images) or self.current_images[index] is None:
                return
                
            # Check if image_label exists
            if not hasattr(self, 'image_label'):
                return
                
            # Convert PIL image to QPixmap
            image = self.current_images[index]
            img_data = image.convert("RGB").tobytes("raw", "RGB")
            qimage = QImage(img_data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            # Scale pixmap to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
            # Display the image
            self.image_label.setPixmap(pixmap)
        except RuntimeError as e:
            # Handle Qt object deleted errors
            print(f"UI update error in display_image (normal if model was changed): {str(e)}")
        except Exception as e:
            # Log other unexpected errors
            print(f"Unexpected error in display_image: {str(e)}")

    def show_previous_image(self):
        """Show the previous image in the batch"""
        try:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.display_image(self.current_image_index)
                
                # Update navigation buttons
                try:
                    if hasattr(self, 'prev_button'):
                        self.prev_button.setEnabled(self.current_image_index > 0)
                    
                    if hasattr(self, 'next_button'):
                        self.next_button.setEnabled(True)  # There's always a next image if we can go back
                    
                    # Update counter
                    if hasattr(self, 'image_counter_label'):
                        self.image_counter_label.setText(f"Image {self.current_image_index+1}/{len(self.current_images)}")
                except RuntimeError:
                    # UI elements were likely deleted during model change
                    print("UI update error: Navigation widgets have been deleted")
        except Exception as e:
            # Log other unexpected errors
            print(f"Unexpected error in show_previous_image: {str(e)}")
            
    def show_next_image(self):
        """Show the next image in the batch"""
        try:
            if self.current_image_index < len(self.current_images) - 1:
                self.current_image_index += 1
                self.display_image(self.current_image_index)
                
                # Update navigation buttons
                try:
                    if hasattr(self, 'prev_button'):
                        self.prev_button.setEnabled(True)  # There's always a previous image if we can go forward
                    
                    if hasattr(self, 'next_button'):
                        self.next_button.setEnabled(self.current_image_index < len(self.current_images) - 1)
                    
                    # Update counter
                    if hasattr(self, 'image_counter_label'):
                        self.image_counter_label.setText(f"Image {self.current_image_index+1}/{len(self.current_images)}")
                except RuntimeError:
                    # UI elements were likely deleted during model change
                    print("UI update error: Navigation widgets have been deleted")
        except Exception as e:
            # Log other unexpected errors
            print(f"Unexpected error in show_next_image: {str(e)}")

    def on_local_model_changed(self, model_name):
        """Handle change in local model selection"""
        # Cancel any running generation thread
        self.stop_generation_if_running()
        
        if not model_name or model_name not in LOCAL_MODELS:
            return

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
        """Refresh the list of Ollama models"""
        # Check if Ollama is available now (may have been started after app launch)
        ollama_available_now = self.ollama_client.is_available()
        
        # If Ollama wasn't available before but is now, we need to create the UI
        if not self.ollama_available and ollama_available_now:
            self.ollama_available = True
            
            # Find the input_layout (parent of the parameters_layout)
            for i in range(self.centralWidget().layout().count()):
                item = self.centralWidget().layout().itemAt(i)
                if isinstance(item, QVBoxLayout) and item != self.centralWidget().layout():
                    input_layout = item
                    break
            
            # Create and add the Ollama UI
            ollama_group = QGroupBox("Ollama Prompt Enhancement")
            ollama_layout = QVBoxLayout()
            
            # Model selection
            ollama_model_layout = QHBoxLayout()
            ollama_model_label = QLabel("Ollama Model:")
            self.ollama_model_combo = QComboBox()
            refresh_ollama_button = QPushButton("Refresh")
            refresh_ollama_button.clicked.connect(self.refresh_ollama_models)
            ollama_model_layout.addWidget(ollama_model_label)
            ollama_model_layout.addWidget(self.ollama_model_combo)
            ollama_model_layout.addWidget(refresh_ollama_button)
            ollama_layout.addLayout(ollama_model_layout)
            
            # Input mode selection
            input_mode_layout = QHBoxLayout()
            self.description_radio = QRadioButton("Description to Tags")
            self.tags_radio = QRadioButton("Enhance Tags")
            self.tags_radio.setChecked(True)  # Default to tag enhancement
            input_mode_layout.addWidget(self.description_radio)
            input_mode_layout.addWidget(self.tags_radio)
            ollama_layout.addLayout(input_mode_layout)
            
            # Enhance button and input for enhancement
            enhance_layout = QHBoxLayout()
            self.enhance_input = QLineEdit()
            self.enhance_input.setPlaceholderText("Enter prompt to enhance")
            self.enhance_button = QPushButton("Enhance Prompt")
            self.enhance_button.clicked.connect(self.enhance_prompt)
            enhance_layout.addWidget(self.enhance_input)
            enhance_layout.addWidget(self.enhance_button)
            ollama_layout.addLayout(enhance_layout)
            
            ollama_group.setLayout(ollama_layout)
            input_layout.addWidget(ollama_group)
            
            # Get models
            self.ollama_models = self.ollama_client.list_models()
            self.ollama_model_combo.addItems(self.ollama_models)
            
            # Show a message
            self.status_label.setText("Ollama connected successfully")
            
        elif self.ollama_available:
            # If Ollama was already available, just refresh the model list
            self.ollama_models = self.ollama_client.list_models()
            current_model = self.ollama_model_combo.currentText()
            
            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItems(self.ollama_models)
            
            # Try to restore previous selection if it exists
            index = self.ollama_model_combo.findText(current_model)
            if index >= 0:
                self.ollama_model_combo.setCurrentIndex(index)
                
            self.status_label.setText("Ollama models refreshed")
    
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
        self.enhance_button.setEnabled(True)
        self.status_label.setText("Error enhancing prompt")
        
        # Show error in a dialog
        QMessageBox.warning(
            self,
            "Ollama Error",
            f"An error occurred while enhancing the prompt:\n{error_message}"
        )

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