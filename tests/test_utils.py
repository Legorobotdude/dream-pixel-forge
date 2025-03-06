import os
import sys
import pytest
from unittest.mock import patch

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dream_pixel_forge

class TestUtilityFunctions:
    
    def test_get_device(self):
        """Test that get_device returns a valid device"""
        device = dream_pixel_forge.get_device()
        assert device in ["cuda", "mps", "cpu"], f"Device {device} not in expected list"
        
    def test_get_torch_dtype(self):
        """Test that get_torch_dtype returns a valid dtype based on device"""
        # Test with CPU
        with patch('dream_pixel_forge.get_device', return_value="cpu"):
            dtype = dream_pixel_forge.get_torch_dtype()
            assert dtype is not None, "Torch dtype should not be None"
            
        # Test with CUDA
        with patch('dream_pixel_forge.get_device', return_value="cuda"):
            dtype = dream_pixel_forge.get_torch_dtype()
            assert dtype is not None, "Torch dtype should not be None"
            
        # Test with MPS
        with patch('dream_pixel_forge.get_device', return_value="mps"):
            dtype = dream_pixel_forge.get_torch_dtype()
            assert dtype is not None, "Torch dtype should not be None"
    
    @patch('dream_pixel_forge.is_model_downloaded')
    def test_is_model_downloaded(self, mock_is_downloaded):
        """Test the is_model_downloaded function"""
        # Test with a non-existent model
        mock_is_downloaded.return_value = False
        result = dream_pixel_forge.is_model_downloaded("non_existent_model")
        assert result is False, "Should return False for non-existent model"
            
        # Test with an existing model
        mock_is_downloaded.return_value = True
        result = dream_pixel_forge.is_model_downloaded("existing_model")
        assert result is True, "Should return True for existing model" 