#!/usr/bin/env python3
"""
Test script to validate the packaging approach.
"""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget

class SimpleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DreamPixelForge Installer Test")
        self.setGeometry(100, 100, 400, 200)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add label
        label = QLabel("DreamPixelForge installation test successful!")
        label.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(label)
        
        # Add button
        button = QPushButton("Exit")
        button.clicked.connect(self.close)
        layout.addWidget(button)

def main():
    app = QApplication(sys.argv)
    window = SimpleApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 