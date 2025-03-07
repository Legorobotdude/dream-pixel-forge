#!/usr/bin/env python3
import os
import sys
import platform
import subprocess
import shutil
import zipfile
import tarfile
from datetime import datetime

VERSION = "1.0.0"  # Update this for new releases

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

def create_distribution_package(platform_name):
    print(f"Creating distribution package for {platform_name}...")
    
    # Create dist directory if it doesn't exist
    os.makedirs("dist", exist_ok=True)
    
    # Get the version and date for the filename
    date_str = datetime.now().strftime("%Y%m%d")
    dist_name = f"DreamPixelForge-{VERSION}-{date_str}-{platform_name}"
    
    # Create a temporary directory for packaging
    temp_dir = os.path.join("dist", dist_name)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Copy the executable and supporting files
    if platform_name == "macos":
        # For macOS, copy the .app bundle
        app_path = os.path.join("dist", "DreamPixelForge.app")
        if os.path.exists(app_path):
            shutil.copytree(app_path, os.path.join(temp_dir, "DreamPixelForge.app"))
    else:
        # For Windows and Linux, copy the executable and supporting files
        dist_path = os.path.join("dist", "DreamPixelForge")
        if os.path.exists(dist_path):
            shutil.copytree(dist_path, os.path.join(temp_dir, "DreamPixelForge"))
    
    # Copy README and other documentation
    for file in ["README.md", "LICENSE"]:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(temp_dir, file))
    
    # Copy the installer script
    shutil.copy2("installer.py", os.path.join(temp_dir, "installer.py"))
    
    # Create a compressed archive
    if platform_name == "windows":
        # Create ZIP file for Windows
        zip_path = os.path.join("dist", f"{dist_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        print(f"Created distribution package: {zip_path}")
    else:
        # Create tarball for macOS and Linux
        tar_path = os.path.join("dist", f"{dist_name}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=dist_name)
        print(f"Created distribution package: {tar_path}")
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    return True

def build_application():
    platform_name = get_platform()
    if not platform_name:
        return False
    
    print(f"Building DreamPixelForge for {platform_name}...")
    
    # First, make sure PyInstaller is installed
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Check if we have a spec file to use
    if os.path.exists("DreamPixelForge.spec"):
        # Build using the spec file
        cmd = [sys.executable, "-m", "PyInstaller", "DreamPixelForge.spec", "--clean"]
    else:
        # Fall back to using the build.py script
        cmd = [sys.executable, "build.py"]
    
    # Run the build command
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Build failed!")
        return False
    
    # Create a distribution package for the current platform
    return create_distribution_package(platform_name)

if __name__ == "__main__":
    if build_application():
        print("\nBuild and packaging completed successfully!")
        sys.exit(0)
    else:
        print("\nBuild or packaging failed.")
        sys.exit(1) 