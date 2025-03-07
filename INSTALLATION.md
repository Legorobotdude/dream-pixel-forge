# DreamPixelForge Installation Guide

This document provides instructions on how to install DreamPixelForge on different operating systems.

## Download

First, download the appropriate package for your operating system from the release page:

- **Windows**: Download `DreamPixelForge-[version]-[date]-windows.zip`
- **macOS**: Download `DreamPixelForge-[version]-[date]-macos.tar.gz`
- **Linux**: Download `DreamPixelForge-[version]-[date]-linux.tar.gz`

## Installation

### Windows

1. Extract the ZIP file to a temporary location
2. Run the `installer.py` script with Python:
   ```
   python3 installer.py
   ```
3. By default, the application will be installed to `C:\Program Files\DreamPixelForge`
4. A desktop shortcut will be created automatically

#### Alternative Manual Installation
1. Extract the ZIP file to any location of your choice
2. Run the application directly by double-clicking `DreamPixelForge.exe`

### macOS

1. Extract the TAR.GZ file to a temporary location
2. Run the installer script:
   ```
   python3 installer.py
   ```
3. By default, the application will be installed to `~/Applications/`
4. A desktop shortcut will be created automatically

#### Alternative Manual Installation
1. Extract the TAR.GZ file
2. Drag the `DreamPixelForge.app` to your Applications folder

### Linux

1. Extract the TAR.GZ file to a temporary location
2. Run the installer script:
   ```
   python3 installer.py
   ```
3. By default, the application will be installed to `~/.local/bin/`
4. A desktop shortcut will be created automatically

#### Alternative Manual Installation
1. Extract the TAR.GZ file to a location of your choice
2. Run the application directly:
   ```
   ./DreamPixelForge/DreamPixelForge
   ```

## Command-line Options

The installer script supports the following command-line options:

- `--install-dir [PATH]`: Specify a custom installation directory
- `--no-shortcut`: Skip creating a desktop shortcut

Example:
```
python3 installer.py --install-dir /custom/path --no-shortcut
```

## Troubleshooting

### Missing Dependencies
If you encounter any missing dependencies when running the application, make sure you have the following installed:

- Python 3.8 or newer (required only for the installer)
- PyQt6 (typically bundled, but may need to be installed separately on some Linux distributions)

### Windows SmartScreen Warning
On Windows, you may see a SmartScreen warning when running the application for the first time. This is normal for applications that are not digitally signed. You can bypass this by clicking "More info" and then "Run anyway".

### macOS Security Warning
On macOS, you may see a message that the application is from an unidentified developer. To allow the application to run:

1. Right-click (or Control-click) on the application
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears

### Linux Package Manager
On Linux, if you prefer to use your system's package manager, you can create a package for your distribution using the source code available at the project's GitHub repository.

## Uninstallation

### Windows
1. Go to "Add or Remove Programs" in the Control Panel
2. Find DreamPixelForge and click "Uninstall"

### macOS
1. Delete the DreamPixelForge.app from your Applications folder
2. Delete any desktop shortcuts

### Linux
1. Delete the DreamPixelForge directory from your installation location
2. Delete any desktop shortcuts

## Support

If you encounter any issues during installation, please report them on our GitHub issue tracker. 