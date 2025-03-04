import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QProgressBar, QFileDialog,
                            QComboBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import traceback

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

class GenerationThread(QThread):
    finished = pyqtSignal(Image.Image)
    progress = pyqtSignal(int, str)  # Now includes status message
    error = pyqtSignal(str)

    def __init__(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height):
        super().__init__()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.pipe = None

    def progress_callback(self, step, timestep, latents):
        if step is not None:
            progress = int((step + 1) / self.num_inference_steps * 100)
            status = f"Generating image... Step {step + 1}/{self.num_inference_steps}"
            self.progress.emit(progress, status)

    def run(self):
        try:
            # Initialize the pipeline
            self.progress.emit(0, "Loading model...")
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA is available, using GPU")
            else:
                print("CUDA is not available, using CPU")
            
            # Initialize pipeline with safety checker
            print("Initializing pipeline...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
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
                image = self.pipe(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    width=self.width,
                    height=self.height,
                    callback=self.progress_callback,
                    callback_steps=1  # Update progress for every step
                ).images[0]

            print("Generation completed successfully")
            self.finished.emit(image)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)
        finally:
            # Clean up
            if self.pipe is not None:
                del self.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DreamPixelForge - Stable Diffusion")
        self.setMinimumSize(800, 600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create input section
        input_layout = QVBoxLayout()
        
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
        self.resolution_combo.addItems(RESOLUTION_PRESETS.keys())
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

    def generate_image(self):
        if not self.prompt_input.text():
            return
            
        self.generate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        
        # Get selected resolution
        selected_resolution = self.resolution_combo.currentText()
        width, height = RESOLUTION_PRESETS[selected_resolution]
        
        self.generation_thread = GenerationThread(
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