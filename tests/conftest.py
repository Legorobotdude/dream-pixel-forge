import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtWidgets import QApplication

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a QApplication instance for tests that need it
@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for PyQt tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    
# Mock for OllamaClient
@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient for testing."""
    mock_client = MagicMock()
    mock_client.list_models.return_value = [
        {"name": "llama2", "modified_at": "2023-01-01T00:00:00Z"},
        {"name": "mistral", "modified_at": "2023-01-01T00:00:00Z"}
    ]
    mock_client.is_available.return_value = True
    mock_client.enhance_prompt.return_value = "Enhanced prompt text"
    return mock_client

# Mock for torch device
@pytest.fixture
def mock_torch_device():
    """Mock torch device to return 'cpu' for tests"""
    with patch('dream_pixel_forge.get_device') as mock_device:
        mock_device.return_value = "cpu"
        yield mock_device

# Mock for PIL Image
@pytest.fixture
def mock_pil_image():
    """Create a mock PIL Image for testing."""
    mock_image = MagicMock()
    mock_image.size = (512, 512)
    return mock_image 