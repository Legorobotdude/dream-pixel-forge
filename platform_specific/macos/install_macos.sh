#!/bin/bash

# DreamPixelForge macOS Installation Script
# This script helps set up DreamPixelForge on macOS systems

# Text styling
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${GREEN}DreamPixelForge macOS Installation Script${NC}"
echo -e "${YELLOW}This script will help you set up DreamPixelForge on your Mac.${NC}"
echo ""

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ -z "$python_version" ]]; then
  echo -e "${YELLOW}Python 3 is not found. Please install Python 3.8 or higher.${NC}"
  echo "Visit https://www.python.org/downloads/ to download and install Python."
  exit 1
fi

echo -e "${GREEN}Python $python_version found.${NC}"

# Check if we're in the right directory
if [[ ! -f "dream_pixel_forge.py" && ! -f "../dream_pixel_forge.py" ]]; then
  echo -e "${YELLOW}Cannot find dream_pixel_forge.py.${NC}"
  echo "Make sure you run this script from the DreamPixelForge directory or platform_specific/macos directory."
  exit 1
fi

# Navigate to root directory if needed
if [[ ! -f "dream_pixel_forge.py" ]]; then
  cd ../..
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt

# Check if PyTorch installed correctly with MPS support
echo -e "${BLUE}Checking PyTorch installation...${NC}"
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'Not supported by this version')"

# Create outputs directory
echo -e "${BLUE}Creating outputs directory...${NC}"
mkdir -p outputs

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}To run DreamPixelForge:${NC}"
echo "1. Activate the virtual environment: ${BOLD}source venv/bin/activate${NC}"
echo "2. Run the application: ${BOLD}python dream_pixel_forge.py${NC}"
echo ""
echo -e "${YELLOW}Tips for macOS:${NC}"
echo "• First runs will download models and will be slower"
echo "• Apple Silicon Macs will use Metal acceleration"
echo "• Use lower resolutions and step counts for faster generation"
echo "• Start with Stable Diffusion 1.5 for best performance"
echo ""
echo -e "${GREEN}Enjoy DreamPixelForge!${NC}" 