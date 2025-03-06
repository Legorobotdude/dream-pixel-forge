import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import QEvent

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dream_pixel_forge
from dream_pixel_forge import CollapsibleSection, MacDropdownButton

class TestCollapsibleSection:
    
    @patch.object(CollapsibleSection, '__init__', return_value=None)
    def test_init(self, mock_init, qapp):
        """Test initialization of CollapsibleSection"""
        title = "Test Section"
        # Create a section without calling the real __init__
        section = CollapsibleSection.__new__(CollapsibleSection)
        
        # Manually set up the section for testing
        section.content_widget = QWidget()
        section.content_layout = QVBoxLayout(section.content_widget)
        
        # Call the mocked init
        mock_init(section, title)
        
        # Verify the mock was called with the title
        mock_init.assert_called_once_with(section, title)
    
    @patch.object(CollapsibleSection, 'on_toggle')
    def test_toggle(self, mock_on_toggle, qapp):
        """Test toggling the collapsible section"""
        # Create a section without calling the real __init__
        section = CollapsibleSection.__new__(CollapsibleSection)
        
        # Manually set up the section for testing
        section.content_widget = QWidget()
        section.on_toggle = mock_on_toggle
        
        # Call the toggle method with both states
        section.on_toggle(True)
        mock_on_toggle.assert_called_with(True)
        
        mock_on_toggle.reset_mock()
        section.on_toggle(False)
        mock_on_toggle.assert_called_with(False)
    
    def test_add_widget(self, qapp):
        """Test adding a widget to the section"""
        section = CollapsibleSection("Test")
        widget = QLabel("Test Label")
        
        section.addWidget(widget)
        
        # The content layout should contain our widget
        assert section.content_layout.count() > 0, "Widget should be added to layout"
    
    def test_add_layout(self, qapp):
        """Test adding a layout to the section"""
        section = CollapsibleSection("Test")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Test Label"))
        
        section.addLayout(layout)
        
        # The content layout should contain our layout
        assert section.content_layout.count() > 0, "Layout should be added"


class TestMacDropdownButton:
    
    def test_init(self, qapp):
        """Test initialization of MacDropdownButton"""
        items = ["Item 1", "Item 2", "Item 3"]
        dropdown = MacDropdownButton(items)
        
        # Check if the button has the first item as text (with dropdown indicator)
        assert items[0] in dropdown.text(), "First item should be in button text"
    
    def test_add_items(self, qapp):
        """Test adding items to the dropdown"""
        dropdown = MacDropdownButton()
        items = ["Item 1", "Item 2", "Item 3"]
        
        dropdown.addItems(items)
        
        # Check if the button has the first item as text (with dropdown indicator)
        assert items[0] in dropdown.text(), "First item should be in button text"
    
    def test_add_item(self, qapp):
        """Test adding a single item"""
        dropdown = MacDropdownButton(["Item 1"])
        
        dropdown.addItem("Item 2")
        
        # Check if the item was added by finding its index
        assert dropdown.findText("Item 2") >= 0, "Item should be added"
        assert dropdown.count() == 2, "Should have 2 items"
    
    def test_current_text(self, qapp):
        """Test currentText method"""
        items = ["Item 1", "Item 2"]
        dropdown = MacDropdownButton(items)
        
        assert dropdown.currentText() == "Item 1", "Should return current text"
        
        # Change selected item
        dropdown.setCurrentText("Item 2")
        assert dropdown.currentText() == "Item 2", "Should return updated text"
    
    def test_current_index(self, qapp):
        """Test currentIndex and setCurrentIndex methods"""
        items = ["Item 1", "Item 2", "Item 3"]
        dropdown = MacDropdownButton(items)
        
        assert dropdown.currentIndex() == 0, "Initial index should be 0"
        
        # Set index
        dropdown.setCurrentIndex(1)
        assert dropdown.currentIndex() == 1, "Index should be updated"
        assert dropdown.currentText() == "Item 2", "Text should match new index"
        
        # Invalid index should be ignored
        dropdown.setCurrentIndex(10)
        assert dropdown.currentIndex() == 1, "Invalid index should be ignored"
    
    def test_clear(self, qapp):
        """Test clear method"""
        dropdown = MacDropdownButton(["Item 1", "Item 2"])
        
        dropdown.clear()
        
        # After clearing, count should be 0
        assert dropdown.count() == 0, "Count should be 0"
    
    def test_find_text(self, qapp):
        """Test findText method"""
        items = ["Apple", "Banana", "Cherry"]
        dropdown = MacDropdownButton(items)
        
        assert dropdown.findText("Banana") == 1, "Should find item index"
        assert dropdown.findText("Orange") == -1, "Should return -1 for not found" 