import os
import sys
import platform
import subprocess

def main():
    print("Starting build process for DreamPixelForge...")
    
    # Determine platform-specific settings
    system = platform.system()
    is_macos = system == "Darwin"
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    
    # Create basic PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=DreamPixelForge",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--add-data=models:models",
        "--add-data=docs:docs",
    ]
    
    # Add platform-specific resources
    if is_macos:
        cmd.extend([
            "--add-data=platform_specific/macos:platform_specific/macos",
            "--icon=docs/images/app_icon.icns" if os.path.exists("docs/images/app_icon.icns") else "",
        ])
    elif is_windows:
        cmd.extend([
            "--add-data=platform_specific/windows;platform_specific/windows" if os.path.exists("platform_specific/windows") else "",
            "--icon=docs/images/app_icon.ico" if os.path.exists("docs/images/app_icon.ico") else "",
        ])
    elif is_linux:
        cmd.extend([
            "--add-data=platform_specific/linux:platform_specific/linux" if os.path.exists("platform_specific/linux") else "",
        ])
    
    # Add main script
    cmd.append("dream_pixel_forge.py")
    
    # Filter out empty strings
    cmd = [c for c in cmd if c]
    
    # Run PyInstaller
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Build failed with the following error:")
        print(result.stderr)
        return 1
    
    # Copy README and license
    dist_path = os.path.join("dist")
    for file in ["README.md", "LICENSE"]:
        if os.path.exists(file):
            dest = os.path.join(dist_path, file)
            subprocess.run(["cp", file, dest])
            print(f"Copied {file} to {dest}")
    
    print("\nBuild completed successfully!")
    print(f"Executable can be found in the 'dist' directory")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 