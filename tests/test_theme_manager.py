import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dream_pixel_forge
from dream_pixel_forge import ThemeManager

class TestThemeManager:
    
    def test_get_theme(self):
        """Test get_theme method"""
        # Test default theme
        theme = ThemeManager.get_theme()
        assert isinstance(theme, dict), "Theme should be a dictionary"
        
        # Update to match actual implementation which might have nested structure
        assert "colors" in theme, "Theme should have colors dictionary"
        assert "background" in theme.get("colors", {}), "Theme should have background color in colors dict"
        
        # Test specific theme
        theme = ThemeManager.get_theme("light")
        assert theme != ThemeManager.get_theme("dark"), "Light theme should differ from dark theme"
        
        # Test invalid theme (should return dark theme)
        theme = ThemeManager.get_theme("invalid_theme")
        assert theme == ThemeManager.get_theme("dark"), "Invalid theme should default to dark"
    
    def test_get_stylesheet(self):
        """Test get_stylesheet method"""
        # Test default stylesheet
        stylesheet = ThemeManager.get_stylesheet()
        assert isinstance(stylesheet, str), "Stylesheet should be a string"
        assert "QWidget" in stylesheet, "Stylesheet should contain QWidget styles"
        
        # Test specific stylesheet
        stylesheet = ThemeManager.get_stylesheet("light")
        assert stylesheet != ThemeManager.get_stylesheet("dark"), "Light stylesheet should differ from dark"
        
        # Test invalid theme (should return dark stylesheet)
        stylesheet = ThemeManager.get_stylesheet("invalid_theme")
        assert stylesheet == ThemeManager.get_stylesheet("dark"), "Invalid theme should default to dark stylesheet"
    
    def test_apply_theme(self):
        """Test apply_theme method"""
        # Create mock app
        app = MagicMock()
        
        # Apply theme
        ThemeManager.apply_theme(app, "dark")
        
        # Verify app.setStyleSheet was called
        app.setStyleSheet.assert_called_once()
        
        # Test with light theme
        app.reset_mock()
        ThemeManager.apply_theme(app, "light")
        app.setStyleSheet.assert_called_once()
        
        # The apply_theme method might not return anything, so we can't compare return values
        # Instead, check that different stylesheets are applied
        dark_app = MagicMock()
        light_app = MagicMock()
        
        ThemeManager.apply_theme(dark_app, "dark")
        ThemeManager.apply_theme(light_app, "light")
        
        dark_style = dark_app.setStyleSheet.call_args[0][0]
        light_style = light_app.setStyleSheet.call_args[0][0]
        
        assert dark_style != light_style, "Dark and light themes should apply different stylesheets" 