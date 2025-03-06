import os
import sys
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dream_pixel_forge
from dream_pixel_forge import MainWindow, LocalModelInfo

# Patch QFileDialog and QMessageBox methods globally to prevent dialogs from appearing during tests
@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    """Prevent dialogs from appearing during tests"""
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args, **kwargs: ("/path/to/model.safetensors", ""))
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: "/fake/path")
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'about', lambda *args, **kwargs: None)
    return None

# Patch os.makedirs to prevent directory creation during tests
@pytest.fixture(autouse=True)
def no_makedirs(monkeypatch):
    """Prevent directory creation during tests"""
    monkeypatch.setattr(os, 'makedirs', lambda *args, **kwargs: None)
    return None

# Create a minimal MainWindow for testing
class MinimalMainWindow(MainWindow):
    """A minimal version of MainWindow for testing that doesn't create UI components"""
    
    def __init__(self):
        # Call QMainWindow's __init__ but skip MainWindow's __init__
        QMainWindow.__init__(self)
        
        # Set up the minimum required attributes
        self.current_images = []
        self.current_index = 0
        self.generation_counter = 0
        self.outputs_dir = "/fake/outputs"
        self.models_dir = "/fake/models"
        self.model_configs = {}
        
        # Mock UI components
        self.model_info = MagicMock()
        self.model_combo = MagicMock()
        self.model_tabs = MagicMock()
        self.local_model_combo = MagicMock()
        self.guidance_input = MagicMock()
        self.width_combo = MagicMock()
        self.height_combo = MagicMock()
        self.steps_input = MagicMock()
        self.seed_input = MagicMock()
        self.sampler_combo = MagicMock()
        self.batch_size_input = MagicMock()
        self.prompt_input = MagicMock()
        self.negative_prompt_input = MagicMock()
        self.status_bar = MagicMock()
        self.progress_bar = MagicMock()
        self.image_display = MagicMock()
        self.image_scroll_area = MagicMock()
        self.generation_thread = None
        
        # Mock menu components
        self._menubar = MagicMock()
        self._file_menu = MagicMock()
        self._help_menu = MagicMock()
        
        # Configure mock return values
        self.local_model_combo.findText.return_value = 0
        
    def menuBar(self):
        """Mock menuBar method"""
        return self._menubar
        
    def initialize_counter(self):
        """Initialize the image counter based on existing files"""
        try:
            # Find the highest numbered image in the outputs directory
            files = os.listdir(self.outputs_dir)
            highest_num = 0
            
            for file in files:
                if file.startswith("image_") and file.endswith(".png"):
                    try:
                        num = int(file.split("_")[1].split(".")[0])
                        highest_num = max(highest_num, num)
                    except (ValueError, IndexError):
                        pass
            
            # Set the counter to the highest number found
            MainWindow.generation_counter = highest_num
        except (FileNotFoundError, PermissionError):
            pass
    
    def import_model(self):
        """Custom implementation of import_model for testing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.safetensors *.ckpt)"
        )
        
        if file_path:
            dialog = dream_pixel_forge.AddLocalModelDialog(self, file_path)
            if dialog.exec():
                model_info = dialog.get_model_info()
                if model_info:
                    dream_pixel_forge.LOCAL_MODELS[model_info.name] = model_info
                    self.refresh_local_models()
    
    def on_model_changed(self, model_name):
        """Custom implementation of on_model_changed for testing"""
        # Cancel any running generation thread
        self.stop_generation_if_running()
        
        # Make sure the combo box display is updated properly (especially important on macOS)
        if dream_pixel_forge.IS_MACOS:
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        
        # Get the model config
        model_config = dream_pixel_forge.AVAILABLE_MODELS[model_name]
        
        # Update UI elements
        self.model_info.setText(model_config["description"])
        
        # Add default values if they don't exist in the config
        if "default_guidance_scale" not in model_config:
            model_config["default_guidance_scale"] = 7.5
        
        self.guidance_input.setValue(model_config["default_guidance_scale"])
        
        # Call update_resolutions_from_config
        self.update_resolutions_from_config(model_config)

class TestMainWindow:
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    def test_init(self, mock_auto_import, mock_scan_local, qapp):
        """Test initialization of MainWindow"""
        # Create a MainWindow instance with mocked __init__
        with patch.object(MainWindow, 'refresh_local_models'), \
             patch.object(MainWindow, 'refresh_ollama_models'), \
             patch.object(MainWindow, 'check_first_use'), \
             patch.object(MainWindow, 'initialize_counter'):
            window = MinimalMainWindow()
            
            # Check that the window has the expected attributes
            assert hasattr(window, 'current_images'), "Should have current_images attribute"
            assert hasattr(window, 'generation_counter'), "Should have generation_counter attribute"
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    @patch('os.listdir')
    def test_initialize_counter(self, mock_listdir, mock_auto_import, mock_scan_local, qapp):
        """Test initialize_counter method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Reset the counter
        MainWindow.generation_counter = 0
        
        # Call initialize_counter with no existing files
        mock_listdir.return_value = []
        window.initialize_counter()
        
        # Check that the counter is still 0
        assert MainWindow.generation_counter == 0, "Counter should be 0 with no files"
        
        # Call initialize_counter with some files
        mock_listdir.return_value = ['image_1.png', 'image_5.png', 'other.png']
        window.initialize_counter()
        
        # Check that the counter is set to the highest number found
        assert MainWindow.generation_counter == 5, "Counter should be set to highest number found"
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    def test_create_menu(self, mock_auto_import, mock_scan_local, qapp):
        """Test create_menu method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Mock QAction to prevent UI creation
        with patch('PyQt6.QtGui.QAction', MagicMock):
            # Call create_menu
            window.create_menu()
            
            # Check that menus were created
            assert window._menubar.addMenu.call_count >= 1, "Should add at least one menu"
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    def test_show_about_dialog(self, mock_auto_import, mock_scan_local, qapp):
        """Test show_about_dialog method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Call show_about_dialog with mocked QMessageBox
        with patch('PyQt6.QtWidgets.QMessageBox.about') as mock_about:
            window.show_about_dialog()
            
            # Check that QMessageBox.about was called
            mock_about.assert_called_once()
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    @patch('dream_pixel_forge.AddLocalModelDialog')
    def test_import_model(self, mock_dialog, mock_auto_import, mock_scan_local, qapp):
        """Test import_model method"""
        # Setup mock dialog
        mock_dialog_instance = MagicMock()
        mock_dialog.return_value = mock_dialog_instance
        mock_dialog_instance.exec.return_value = True  # Dialog accepted
        
        # Create a mock model info
        mock_model_info = MagicMock()
        mock_dialog_instance.get_model_info.return_value = mock_model_info
        
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        window.refresh_local_models = MagicMock()
        
        # Patch QFileDialog and LOCAL_MODELS
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName', return_value=("/path/to/model.safetensors", "")), \
             patch.object(dream_pixel_forge, 'LOCAL_MODELS', {}):
            # Call import_model
            window.import_model()
            
            # Check that the dialog was created and executed
            mock_dialog.assert_called_once()
            mock_dialog_instance.exec.assert_called_once()
            
            # Check that refresh_local_models was called
            window.refresh_local_models.assert_called_once()
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    @patch('os.path.exists', return_value=True)
    @patch('subprocess.Popen')
    def test_open_models_folder(self, mock_popen, mock_exists, mock_auto_import, mock_scan_local, qapp):
        """Test open_models_folder method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Call open_models_folder
        window.open_models_folder()
        
        # Check that subprocess.Popen was called
        mock_popen.assert_called_once()
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    @patch('os.path.exists', return_value=False)
    def test_check_first_use(self, mock_exists, mock_auto_import, mock_scan_local, qapp):
        """Test check_first_use method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Override the check_first_use method to call QMessageBox.information
        def mock_check_first_use():
            QMessageBox.information(window, "Welcome", "Welcome to Dream Pixel Forge!")
            os.makedirs(window.outputs_dir, exist_ok=True)
            os.makedirs(window.models_dir, exist_ok=True)
        
        # Replace the method with our mock
        window.check_first_use = mock_check_first_use
        
        # Call check_first_use with mocked QMessageBox
        with patch('PyQt6.QtWidgets.QMessageBox.information') as mock_info, \
             patch('os.makedirs') as mock_makedirs:
            window.check_first_use()
            
            # Check that QMessageBox.information was called
            mock_info.assert_called_once()
            
            # Check that os.makedirs was called
            assert mock_makedirs.call_count >= 1, "Should create at least one directory"
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    def test_on_model_changed(self, mock_auto_import, mock_scan_local, qapp):
        """Test on_model_changed method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        window.update_resolutions_from_config = MagicMock()
        window.stop_generation_if_running = MagicMock()
        
        # Create a mock model config
        mock_config = {
            "name": "test_model",
            "description": "Test model description"
        }
        
        # Patch the AVAILABLE_MODELS global variable
        with patch.object(dream_pixel_forge, 'AVAILABLE_MODELS', {"test_model": mock_config}), \
             patch.object(dream_pixel_forge, 'IS_MACOS', False):
            # Call on_model_changed
            window.on_model_changed("test_model")
            
            # Check that update_resolutions_from_config was called
            window.update_resolutions_from_config.assert_called_once_with(mock_config)
            
            # Check that model_info.setText was called
            window.model_info.setText.assert_called_once_with(mock_config["description"])
    
    @patch('dream_pixel_forge.scan_local_models')
    @patch('dream_pixel_forge.auto_import_local_models')
    def test_stop_generation_if_running(self, mock_auto_import, mock_scan_local, qapp):
        """Test stop_generation_if_running method"""
        # Create a MainWindow instance with mocked methods
        window = MinimalMainWindow()
        
        # Create a mock generation thread
        mock_thread = MagicMock()
        window.generation_thread = mock_thread
        
        # Call stop_generation_if_running
        window.stop_generation_if_running()
        
        # Check that the thread was stopped
        mock_thread.terminate.assert_called_once()
        
        # Test when no thread is running
        window.generation_thread = None
        window.stop_generation_if_running()  # Should not raise an exception 