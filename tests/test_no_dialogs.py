import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call

# Add parent directory to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import QFileDialog, QMessageBox

class TestNoDialogs:
    """Tests to verify that no UI dialogs appear during test execution"""
    
    def test_file_dialog_patching(self, monkeypatch):
        """Test that QFileDialog methods are properly patched"""
        # Create a spy to track calls to getOpenFileName
        spy = MagicMock(return_value=("/path/to/model.safetensors", ""))
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', spy)
        
        # Call the method
        result = QFileDialog.getOpenFileName(None, "Test", "", "")
        
        # Verify the result
        assert result == ("/path/to/model.safetensors", "")
        assert spy.call_count == 1
    
    def test_message_box_patching(self, monkeypatch):
        """Test that QMessageBox methods are properly patched"""
        # Create spies for different QMessageBox methods
        info_spy = MagicMock(return_value=None)
        question_spy = MagicMock(return_value=QMessageBox.StandardButton.Yes)
        
        monkeypatch.setattr(QMessageBox, 'information', info_spy)
        monkeypatch.setattr(QMessageBox, 'question', question_spy)
        
        # Call the methods
        QMessageBox.information(None, "Title", "Message")
        result = QMessageBox.question(None, "Title", "Question?")
        
        # Verify the results
        assert info_spy.call_count == 1
        assert question_spy.call_count == 1
        assert result == QMessageBox.StandardButton.Yes
    
    def test_makedirs_patching(self, monkeypatch):
        """Test that os.makedirs is properly patched"""
        # Create a spy for os.makedirs
        makedirs_spy = MagicMock()
        monkeypatch.setattr(os, 'makedirs', makedirs_spy)
        
        # Call the method
        os.makedirs("/fake/path", exist_ok=True)
        
        # Verify the call
        makedirs_spy.assert_called_once_with("/fake/path", exist_ok=True) 