#!/bin/bash
# AMD MONAI Pathology Analysis - Setup Script
# Automated installation for AMD ROCm environment

set -e  # Exit on any error

echo "🔬 AMD MONAI Pathology Analysis - Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems. Detected: $OSTYPE"
    exit 1
fi

# Check Python version
print_header "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $PYTHON_VERSION"
    
    # Check if Python >= 3.10
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
        print_status "Python version is compatible (>=3.10)"
    else
        print_warning "Python 3.10+ recommended for best compatibility"
    fi
else
    print_error "Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check for ROCm
print_header "Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    print_status "ROCm found - checking GPU status..."
    rocm-smi --showproductname 2>/dev/null || print_warning "ROCm installed but no GPU detected"
else
    print_warning "ROCm not found. Please install ROCm 6.0+ for GPU acceleration."
    echo "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
fi

# Check for required system packages
print_header "Checking system dependencies..."
missing_packages=()

for package in libopenslide0 python3-pip python3-venv; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        missing_packages+=($package)
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    print_warning "Missing system packages: ${missing_packages[*]}"
    read -p "Install missing packages? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing system packages..."
        sudo apt update
        sudo apt install -y "${missing_packages[@]}"
        
        # Install ROCm development packages if available
        print_status "Attempting to install ROCm development packages..."
        sudo apt install -y rocthrust-dev hipcub hipblas hipblas-dev hipfft hipsparse hiprand rocsolver rocrand-dev rocjpeg 2>/dev/null || print_warning "Some ROCm packages not available - manual ROCm installation may be required"
    fi
fi

# Create virtual environment
print_header "Setting up Python virtual environment..."
VENV_NAME="monai_pathology_env"

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment $VENV_NAME already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        print_status "Removed existing virtual environment"
    else
        print_status "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    print_status "Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_header "Installing Python packages..."

# Check for ROCm and install PyTorch accordingly
if command -v rocm-smi &> /dev/null; then
    print_status "ROCm detected - installing PyTorch with ROCm support first..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
    
    print_status "Installing other requirements..."
    pip install fastapi uvicorn[standard] python-multipart
    pip install monai\>=1.5.0
    pip install Pillow numpy opencv-python-headless
    pip install scipy scikit-image matplotlib pandas
    pip install pytest jupyter notebook tqdm pydantic
else
    print_warning "ROCm not detected - using standard requirements..."
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
fi

# Install hipCIM
print_header "Installing hipCIM (AMD's cuCIM port)..."
print_status "Installing amd-hipcim from AMD PyPI..."
pip install amd-hipcim --extra-index-url=https://pypi.amd.com/simple

# Verification
print_header "Verifying installation..."

print_status "Testing hipCIM..."
python3 -c "
try:
    from cucim import CuImage
    print('✅ hipCIM imported successfully')
except ImportError as e:
    print('❌ hipCIM import failed:', e)
    exit(1)
" || print_warning "hipCIM verification failed"

print_status "Testing MONAI..."
python3 -c "
try:
    import monai
    print('✅ MONAI imported successfully - Version:', monai.__version__)
except ImportError as e:
    print('❌ MONAI import failed:', e)
    exit(1)
"

print_status "Testing PyTorch..."
python3 -c "
import torch
print('✅ PyTorch imported - Version:', torch.__version__)
print('ROCm GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name())
else:
    print('No GPU detected - CPU mode will be used')
"

# Final instructions
print_header "Setup Complete!"
echo ""
print_status "To activate the environment in future sessions:"
echo "source $VENV_NAME/bin/activate"
echo ""
print_status "To run the application:"
echo "uvicorn chatbot_monai_medical:app --host 0.0.0.0 --port 8000 --reload"
echo ""
print_status "Then open your browser to: http://localhost:8000"
echo ""
print_status "For troubleshooting, see INSTALLATION.md"

# Check for sample data
if [ ! -f "data/sample_wsi.svs" ]; then
    print_warning "No sample WSI file found at data/sample_wsi.svs"
    print_status "Place your .svs files in the data/ directory to get started"
fi

echo ""
echo "🎉 Setup complete! Ready to analyze pathology slides with AMD ROCm acceleration."