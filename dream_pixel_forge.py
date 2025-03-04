import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
                            QComboBox, QMessageBox, QGroupBox, QRadioButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, KandinskyV22Pipeline
import torch
from PIL import Image
import io
import traceback
from huggingface_hub import scan_cache_dir, HfFolder, model_info
import time
import requests
import json
import random

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

# Default negative prompts that work well with SD v1.5
DEFAULT_NEGATIVE_PROMPT = "ugly, blurry, poor quality, distorted, deformed, disfigured, poorly drawn face, poorly drawn hands, poorly drawn feet, poorly drawn legs, deformed, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad proportions, gross proportions, duplicate, morbid, mutilated, extra fingers, fused fingers, too many fingers, long neck, bad composition, bad perspective, bad lighting, watermark, signature, text, logo, banner, extra digits, mutated hands and fingers, poorly drawn hands, poorly drawn face, poorly drawn feet, poorly drawn legs, poorly drawn limbs, poorly drawn anatomy, wrong anatomy, incorrect anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, ugly, disgusting, bad proportions, gross proportions, duplicate, morbid, mutilated, extra fingers, fused fingers, too many fingers, long neck, bad composition, bad perspective, bad lighting, watermark, signature, text, logo, banner, extra digits"

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
        "size_gb": 4.0
    },
    "Stable Diffusion 2.1": {
        "model_id": "stabilityai/stable-diffusion-2-1",
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 7.5,
        "description": "Improved version with better quality and consistency",
        "size_gb": 4.2
    },
    "Stable Diffusion XL": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline": StableDiffusionXLPipeline,
        "resolution_presets": SDXL_RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 9.0,
        "description": "Larger model with higher quality outputs (needs more VRAM)",
        "size_gb": 6.5
    },
    "Dreamlike Diffusion": {
        "model_id": "dreamlike-art/dreamlike-diffusion-1.0",
        "pipeline": StableDiffusionPipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 8.0,
        "description": "Artistic model that creates dreamlike, surreal images",
        "size_gb": 4.0
    },
    "Kandinsky 2.2": {
        "model_id": "kandinsky-community/kandinsky-2-2-decoder",
        "pipeline": KandinskyV22Pipeline,
        "resolution_presets": RESOLUTION_PRESETS,
        "supports_negative_prompt": True,
        "default_guidance_scale": 8.0,
        "description": "Russian alternative to SD with unique artistic style",
        "size_gb": 4.5
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
            status = f"Generating image... Step {step + 1}/{self.num_inference_steps}"
            self.progress.emit(progress, status)

    def run(self):
        try:
            # Initialize the pipeline
            self.progress.emit(0, "Loading model...")
            model_id = self.model_config["model_id"]
            pipeline_class = self.model_config["pipeline"]
            size_gb = self.model_config.get("size_gb", 4.0)
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA is available, using GPU")
                print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            else:
                print("CUDA is not available, using CPU")
            
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
            self.pipe = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Stop the download tracker if it's running
            if self.download_tracker:
                self.download_tracker.stop()
                self.download_tracker.wait()
                self.download_tracker = None
                
                # Let the user know download has completed
                self.progress.emit(0, f"Model {model_id} downloaded successfully! Moving to GPU...")
                QApplication.processEvents()  # Make sure the message is displayed
            
            if torch.cuda.is_available():
                self.progress.emit(0, "Moving model to GPU...")
                print("Moving model to GPU...")
                self.pipe = self.pipe.to("cuda")
                # Try to enable memory efficient attention if available
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
                
                # Set generator for reproducible results if seed is provided
                if i == 0 and self.seed is not None:
                    # For the first image, use the provided seed
                    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                    generator.manual_seed(self.seed)
                    current_seed = self.seed
                    print(f"Using seed for image {i+1}: {current_seed}")
                else:
                    # For subsequent images or if no seed was provided, use random seeds
                    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
                    # Generate a seed that fits in a 32-bit integer to avoid overflow
                    random_seed = random.randint(0, 2147483647)
                    generator.manual_seed(random_seed)
                    current_seed = random_seed
                    print(f"Using random seed for image {i+1}: {current_seed}")
                
                self.generated_seeds.append(current_seed)
                
                # Set the scheduler (sampler) if specified
                if self.sampler and hasattr(self.pipe, "scheduler"):
                    try:
                        # Import necessary samplers dynamically to avoid bloating the imports
                        from diffusers import (
                            DDIMScheduler, DPMSolverMultistepScheduler,
                            EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                            LMSDiscreteScheduler, HeunDiscreteScheduler,
                            KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                            DPMSolverSinglestepScheduler
                        )
                        
                        # Map the sampler name to the appropriate scheduler class
                        schedulers = {
                            "ddim": DDIMScheduler,
                            "dpmpp_2m": DPMSolverMultistepScheduler,
                            "euler_ancestral": EulerAncestralDiscreteScheduler,
                            "euler": EulerDiscreteScheduler,
                            "lms": LMSDiscreteScheduler,
                            "heun": HeunDiscreteScheduler,
                            "dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
                            "dpm_2": KDPM2DiscreteScheduler,
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
                    # Different models might have slightly different APIs
                    if isinstance(self.pipe, StableDiffusionXLPipeline):
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
                
                generated_images.append(image)

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DreamPixelForge - Text to Image")
        self.setMinimumSize(800, 600)
        
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
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS.keys())
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        
        # Add a model info label
        self.model_info = QLabel(AVAILABLE_MODELS["Stable Diffusion 1.5"]["description"])
        self.model_info.setWordWrap(True)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
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
        self.neg_prompt_input.setText(DEFAULT_NEGATIVE_PROMPT)
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
        self.update_resolutions()
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
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        self.current_images = []
        self.generation_thread = None
        
        # Check if first time use
        self.check_first_use()

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
                "- Stable Diffusion XL: ~6.5GB\n\n"
                "Models are downloaded only once and cached for future use."
            )

    def on_model_changed(self, model_name):
        """Update UI when model selection changes"""
        model_config = AVAILABLE_MODELS[model_name]
        self.model_info.setText(model_config["description"])
        self.guidance_input.setValue(model_config["default_guidance_scale"])
        self.update_resolutions()
        
        # Enable/disable negative prompt based on model support
        self.neg_prompt_input.setEnabled(model_config["supports_negative_prompt"])
        
        # Show download size information if model isn't downloaded yet
        if not is_model_downloaded(model_config["model_id"]):
            size_gb = model_config.get("size_gb", 4.0)
            self.status_label.setText(f"Note: {model_name} (~{size_gb:.1f}GB) will be downloaded on first use")
        else:
            # Reset status label if model is already downloaded
            self.status_label.setText("Ready")

    def update_resolutions(self):
        """Update the resolution presets based on selected model"""
        current_model = self.model_combo.currentText()
        resolution_presets = AVAILABLE_MODELS[current_model]["resolution_presets"]
        
        # Remember current selection if possible
        current_selection = self.resolution_combo.currentText()
        
        self.resolution_combo.clear()
        self.resolution_combo.addItems(resolution_presets.keys())
        
        # Try to restore previous selection if it exists in the new list
        index = self.resolution_combo.findText(current_selection)
        if index >= 0:
            self.resolution_combo.setCurrentIndex(index)

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
        
        # Get current model configuration
        current_model = self.model_combo.currentText()
        model_config = AVAILABLE_MODELS[current_model]
        
        # Get selected resolution
        selected_resolution = self.resolution_combo.currentText()
        resolution_presets = model_config["resolution_presets"]
        width, height = resolution_presets[selected_resolution]
        
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
        
        self.generation_thread = GenerationThread(
            model_config,
            self.prompt_input.text(),
            self.neg_prompt_input.text(),
            self.steps_input.value(),
            self.guidance_input.value(),
            width,
            height,
            seed_value,
            sampler_id,
            self.batch_input.value()
        )
        
        self.generation_thread.finished.connect(self.handle_generated_images)
        self.generation_thread.progress.connect(self.handle_progress)
        self.generation_thread.error.connect(self.handle_error)
        self.generation_thread.start()

    def handle_progress(self, progress, status):
        """Handle progress updates from the generation thread"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
        # Force the UI to refresh immediately, particularly important for download messages
        QApplication.processEvents()

    def handle_generated_images(self, images):
        self.current_images = images
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Generated {len(images)} images")
        self.generate_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Update save button text based on batch size
        if len(images) > 1:
            self.save_button.setText("Save Images")
        else:
            self.save_button.setText("Save Image")
        
        # Update the display to show first image
        if images:
            # Update the seed display
            if self.generation_thread and self.generation_thread.generated_seeds:
                seed_str = ", ".join(map(str, self.generation_thread.generated_seeds[:3]))
                if len(self.generation_thread.generated_seeds) > 3:
                    seed_str += f", ... ({len(self.generation_thread.generated_seeds)} total)"
                self.status_label.setText(f"Generated {len(images)} images with seeds: {seed_str}")
                
                # If the user set a specific seed for the first image, show it in the input
                original_seed_setting = self.seed_input.value()
                if original_seed_setting != -1:
                    # Keep the first seed in the input box, which was the user's specified seed
                    self.seed_input.setValue(self.generation_thread.generated_seeds[0])
            
            # Show the first image
            self.current_image_index = 0
            self.display_image(0)
            
            # If we have multiple images, add next/prev buttons
            if len(images) > 1 and not hasattr(self, 'image_nav_layout'):
                # Create navigation buttons for browsing images
                self.image_nav_layout = QHBoxLayout()
                
                self.prev_button = QPushButton("Previous")
                self.prev_button.clicked.connect(self.show_previous_image)
                self.prev_button.setEnabled(False)  # Disabled at first image
                
                self.image_counter_label = QLabel(f"Image 1/{len(images)}")
                self.image_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                self.next_button = QPushButton("Next")
                self.next_button.clicked.connect(self.show_next_image)
                
                self.image_nav_layout.addWidget(self.prev_button)
                self.image_nav_layout.addWidget(self.image_counter_label)
                self.image_nav_layout.addWidget(self.next_button)
                
                # Find the layout that contains the image label
                layout = self.centralWidget().layout()
                layout.insertLayout(layout.count() - 1, self.image_nav_layout)  # Insert before the save button
    
    def display_image(self, index):
        """Display an image from the batch at the specified index"""
        if not self.current_images or index < 0 or index >= len(self.current_images):
            return
            
        image = self.current_images[index]
        
        # Convert PIL image to QPixmap
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        
        # Update navigation controls if they exist
        if hasattr(self, 'image_counter_label'):
            self.image_counter_label.setText(f"Image {index + 1}/{len(self.current_images)}")
            self.prev_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self.current_images) - 1)
    
    def show_next_image(self):
        """Show the next image in the batch"""
        if hasattr(self, 'current_image_index') and self.current_images:
            next_index = min(self.current_image_index + 1, len(self.current_images) - 1)
            self.current_image_index = next_index
            self.display_image(next_index)
    
    def show_previous_image(self):
        """Show the previous image in the batch"""
        if hasattr(self, 'current_image_index') and self.current_images:
            prev_index = max(self.current_image_index - 1, 0)
            self.current_image_index = prev_index
            self.display_image(prev_index)

    def handle_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error occurred - Ready for new generation")
        self.generate_button.setEnabled(True)
        
        # Show error in a dialog
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during image generation:\n{error_message}"
        )

    def save_image(self):
        if not self.current_images:
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
                    
                    # Save each image with index and seed
                    for i, image in enumerate(images_to_save):
                        seed = self.generation_thread.generated_seeds[i] if hasattr(self.generation_thread, 'generated_seeds') and i < len(self.generation_thread.generated_seeds) else "unknown"
                        file_path = os.path.join(save_dir, f"{base_name}_{i+1}_seed{seed}.png")
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

    def refresh_ollama_models(self):
        """Refresh the list of Ollama models"""
        # First check if Ollama is available now (may have been started after app launch)
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 