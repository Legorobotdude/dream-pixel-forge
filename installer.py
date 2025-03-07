#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import shutil
import tempfile
from pathlib import Path
import argparse

def get_platform():
    system = platform.system()
    if system == "Darwin":
        return "macos"
    elif system == "Windows":
        return "windows"
    elif system == "Linux":
        return "linux"
    else:
        print(f"Unsupported platform: {system}")
        return None

def create_desktop_shortcut(app_path, desktop_path):
    platform_type = get_platform()
    
    if platform_type == "macos":
        # Create a symlink on macOS
        shortcut_path = os.path.join(desktop_path, "DreamPixelForge.app")
        if os.path.exists(shortcut_path):
            os.remove(shortcut_path)
        os.symlink(app_path, shortcut_path)
        return True
    
    elif platform_type == "windows":
        # Create a .lnk file on Windows
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(os.path.join(desktop_path, "DreamPixelForge.lnk"))
            shortcut.Targetpath = app_path
            shortcut.WorkingDirectory = os.path.dirname(app_path)
            shortcut.IconLocation = app_path
            shortcut.save()
            return True
        except ImportError:
            print("Could not create Windows shortcut: win32com module not available")
            return False
    
    elif platform_type == "linux":
        # Create a .desktop file on Linux
        desktop_file = os.path.join(desktop_path, "DreamPixelForge.desktop")
        with open(desktop_file, "w") as f:
            f.write(f"""[Desktop Entry]
Type=Application
Name=DreamPixelForge
Exec="{app_path}"
Icon={os.path.join(os.path.dirname(app_path), "icons", "app_icon.png")}
Terminal=false
Categories=Graphics;
""")
        os.chmod(desktop_file, 0o755)
        return True
    
    return False

def install_app():
    platform_type = get_platform()
    if not platform_type:
        return False
    
    # Determine installation path
    parser = argparse.ArgumentParser(description='Install DreamPixelForge')
    parser.add_argument('--install-dir', help='Installation directory')
    parser.add_argument('--no-shortcut', action='store_true', help='Skip creating desktop shortcut')
    args = parser.parse_args()
    
    if args.install_dir:
        install_dir = args.install_dir
    else:
        if platform_type == "macos":
            install_dir = os.path.expanduser("~/Applications")
        elif platform_type == "windows":
            install_dir = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "DreamPixelForge")
        elif platform_type == "linux":
            install_dir = os.path.expanduser("~/.local/bin")
    
    print(f"Installing DreamPixelForge to: {install_dir}")
    
    # Create installation directory if it doesn't exist
    os.makedirs(install_dir, exist_ok=True)
    
    # Find the executable in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform_type == "macos":
        app_path = os.path.join(current_dir, "DreamPixelForge.app")
        dest_path = os.path.join(install_dir, "DreamPixelForge.app")
    elif platform_type == "windows":
        app_path = os.path.join(current_dir, "DreamPixelForge.exe")
        dest_path = os.path.join(install_dir, "DreamPixelForge.exe")
    else:  # linux
        app_path = os.path.join(current_dir, "DreamPixelForge")
        dest_path = os.path.join(install_dir, "DreamPixelForge")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find application at {app_path}")
        return False
        
    # Copy the application to the installation directory
    if os.path.exists(dest_path):
        # Remove existing installation
        if platform_type == "macos":
            shutil.rmtree(dest_path, ignore_errors=True)
        else:
            try:
                os.remove(dest_path)
            except:
                print(f"Warning: Could not remove existing file at {dest_path}")
    
    # Copy the application
    try:
        if platform_type == "macos":
            shutil.copytree(app_path, dest_path)
        else:
            shutil.copy2(app_path, dest_path)
        
        # Also copy any support files/directories
        for item in os.listdir(current_dir):
            if item not in ["DreamPixelForge.app", "DreamPixelForge.exe", "DreamPixelForge", "installer.py"]:
                src = os.path.join(current_dir, item)
                dst = os.path.join(install_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    except Exception as e:
        print(f"Error during installation: {e}")
        return False
    
    # Create desktop shortcut if requested
    if not args.no_shortcut:
        desktop_path = os.path.expanduser("~/Desktop")
        if create_desktop_shortcut(dest_path, desktop_path):
            print("Created desktop shortcut")
        else:
            print("Could not create desktop shortcut")
    
    print("\nInstallation complete!")
    print(f"You can now run DreamPixelForge from {dest_path}")
    return True

if __name__ == "__main__":
    if install_app():
        sys.exit(0)
    else:
        sys.exit(1) 