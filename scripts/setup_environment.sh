#!/bin/bash
# BlindSpot Environment Setup Script
# Sets up Python environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "BlindSpot Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓ Python $python_version found${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required, found $python_version${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Additional dependencies for frame selector
echo ""
echo "Installing additional ML dependencies..."
pip install scikit-learn imagehash > /dev/null
echo -e "${GREEN}✓ Additional dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw/{coco,sun_rgbd,custom}/{images,annotations}
mkdir -p data/processed/{images,labels}/{train,val,test}
mkdir -p models/{pretrained,checkpoints,best}
echo -e "${GREEN}✓ Directories created${NC}"

# Make scripts executable
echo ""
echo "Setting script permissions..."
chmod +x data_collection/*.py
chmod +x scripts/*.sh
echo -e "${GREEN}✓ Scripts are executable${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download COCO data: python data_collection/download_coco.py"
echo "  3. Read documentation: docs/DATA_COLLECTION.md"
echo ""
echo "To deactivate environment: deactivate"
echo ""
