import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
                            QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, KandinskyV22Pipeline
import torch
from PIL import Image
import io
import traceback
from huggingface_hub import scan_cache_dir, HfFolder, model_info
import time

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
            time.sleep(2)
            
    def stop(self):
        self.running = False

class GenerationThread(QThread):
    finished = pyqtSignal(Image.Image)
    progress = pyqtSignal(int, str)  # Now includes status message
    error = pyqtSignal(str)

    def __init__(self, model_config, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height):
        super().__init__()
        self.model_config = model_config
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.pipe = None
        self.download_tracker = None

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
            
            # Initialize pipeline with safety checker
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

            # Generate the image
            self.progress.emit(0, "Starting generation...")
            print(f"Starting generation with prompt: {self.prompt}")
            print(f"Negative prompt: {self.negative_prompt}")
            print(f"Steps: {self.num_inference_steps}, Guidance: {self.guidance_scale}")
            print(f"Resolution: {self.width}x{self.height}")
            
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
                        callback_steps=1
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
                        callback_steps=1
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
                        callback_steps=1
                    ).images[0]

            print("Generation completed successfully")
            self.finished.emit(image)
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
        
        input_layout.addLayout(params_layout)
        layout.addLayout(input_layout)
        
        # Generate button
        self.generate_button = QPushButton("Generate Image")
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
        
        # Save button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        self.current_image = None
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
        
        self.generation_thread = GenerationThread(
            model_config,
            self.prompt_input.text(),
            self.neg_prompt_input.text(),
            self.steps_input.value(),
            self.guidance_input.value(),
            width,
            height
        )
        
        self.generation_thread.finished.connect(self.handle_generated_image)
        self.generation_thread.progress.connect(self.handle_progress)
        self.generation_thread.error.connect(self.handle_error)
        self.generation_thread.start()

    def handle_progress(self, progress, status):
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def handle_generated_image(self, image):
        self.current_image = image
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.generate_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Convert PIL image to QPixmap
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        qimage = QImage.fromData(buffer.getvalue())
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def handle_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error occurred")
        self.generate_button.setEnabled(True)
        
        # Show error in a dialog
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during image generation:\n{error_message}"
        )

    def save_image(self):
        if not self.current_image:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            self.current_image.save(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 