#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) Test Script for macOS
This script checks if PyTorch can use Metal acceleration on macOS.
"""

import sys
import platform
import os

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"System: {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print()

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if MPS is available (macOS with Metal)
    if hasattr(torch.backends, 'mps'):
        is_mps_available = torch.backends.mps.is_available()
        is_mps_built = torch.backends.mps.is_built()
        print(f"MPS built: {is_mps_built}")
        print(f"MPS available: {is_mps_available}")
        
        if is_mps_available:
            print("✅ MPS acceleration is available! Metal will be used for acceleration.")
            
            # Test creating a tensor on MPS device
            try:
                device = torch.device("mps")
                x = torch.ones(5, device=device)
                print(f"Test tensor created on MPS device: {x}")
                print("✅ Successfully created tensor on MPS device")
            except Exception as e:
                print(f"❌ Error when creating tensor on MPS device: {e}")
        else:
            print("❌ MPS is not available on this system")
            if platform.system() == 'Darwin':
                print("This may be because you're running on an Intel Mac or using an older macOS version")
                print("Metal acceleration requires macOS 12.3+ and an Apple Silicon Mac")
            else:
                print("MPS is only available on macOS with Apple Silicon chips")
    else:
        print("❌ This PyTorch version doesn't support MPS (Metal)")
        print("Please ensure you're using PyTorch 1.12 or later")
        
    # Check if CUDA is available (unlikely on macOS)
    if torch.cuda.is_available():
        print(f"CUDA available: Yes (version {torch.version.cuda})")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No")
        
except ImportError:
    print("❌ PyTorch is not installed")
    print("Please install PyTorch with: pip install torch")

print()
print("To use Metal acceleration in DreamPixelForge:")
print("1. Make sure you have an Apple Silicon Mac (M1/M2/M3)")
print("2. Make sure you're running macOS 12.3 or later")
print("3. Use PyTorch 1.12 or later")
print("4. Run DreamPixelForge normally - Metal will be used automatically if available") 