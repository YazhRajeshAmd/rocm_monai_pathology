# AMD MONAI Pathology Analysis - Installation Guide

## Prerequisites

This application requires AMD ROCm for GPU acceleration and hipCIM for whole slide image processing.

### System Requirements
- AMD ROCm-compatible GPU (MI200/MI300 series recommended)
- Ubuntu 22.04 or compatible Linux distribution
- Python 3.10 or higher
- ROCm 6.0+ installed

## Installation Steps

### 1. System Dependencies (Ubuntu/Debian)

```bash
# Update system packages
sudo apt update

# Install required system libraries
sudo apt install -y lsb-release software-properties-common libopenslide0 python3.10-venv rocjpeg

# Install ROCm development libraries
sudo apt install -y rocthrust-dev hipcub hipblas \
                    hipblas-dev hipfft hipsparse \
                    hiprand rocsolver rocrand-dev

# Upgrade pip
pip install --upgrade pip
```

### 2. Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv monai_pathology_env
source monai_pathology_env/bin/activate

# Ensure pip is up to date
pip install --upgrade pip
```

### 3. Install Base Requirements

```bash
# Install main requirements
pip install -r requirements.txt
```

### 4. Install hipCIM (ROCm cuCIM Port)

hipCIM is AMD's ROCm port of cuCIM for high-performance medical image processing.

```bash
# Install hipCIM from AMD PyPI
pip install amd-hipcim --extra-index-url=https://pypi.amd.com/simple
```

**Note**: hipCIM requires ROCm to be properly installed and configured on your system.

### 5. Verify Installation

```bash
# Test hipCIM installation
python3 -c "
try:
    from cucim import CuImage
    print('✅ hipCIM installed successfully')
except ImportError as e:
    print('❌ hipCIM installation failed:', e)
"

# Test MONAI installation  
python3 -c "
try:
    import monai
    print('✅ MONAI installed successfully - Version:', monai.__version__)
except ImportError as e:
    print('❌ MONAI installation failed:', e)
"

# Test PyTorch ROCm support
python3 -c "
import torch
print('✅ PyTorch installed - Version:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name())
"
```

## Running the Application

### 1. Prepare Data
Place your SVS (Whole Slide Image) files in the `data/` directory:
```bash
# Example: Copy your SVS file
cp your_slide.svs data/sample_wsi.svs
```

### 2. Start the Server
```bash
# Activate virtual environment (if not already active)
source monai_pathology_env/bin/activate

# Run the application
python chatbot_monai_medical.py
```

### 3. Access the Web Interface
Open your browser and navigate to: `http://localhost:8000`

## Docker Installation (Alternative)

For a containerized setup with ROCm support:

```bash
# Pull ROCm development image
docker pull rocm/dev-ubuntu-22.04:6.4.1-complete

# Run container with GPU access
docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true \
     --shm-size=128GB --network=host --device=/dev/kfd \
     --device=/dev/dri --group-add video -it \
     -v $HOME:$HOME --name rocm_monai \
     rocm/dev-ubuntu-22.04:6.4.1-complete

# Inside container, follow steps 1-5 above
```

## Troubleshooting

### hipCIM Issues
1. **Import Error**: Ensure ROCm is properly installed and `rocm-smi` works
2. **GPU Not Found**: Verify your AMD GPU is ROCm-compatible
3. **Permission Error**: Add your user to the `video` group: `sudo usermod -a -G video $USER`

### MONAI Issues
1. **Model Loading**: Ensure you have internet connection for first-time model download
2. **Memory Error**: Reduce batch size or image resolution in the code

### PyTorch ROCm Issues
1. **CUDA Not Available**: Install PyTorch with ROCm support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
   ```

## Performance Tips

1. **GPU Memory**: Monitor GPU memory usage with `rocm-smi`
2. **Image Size**: Large SVS files (>100MB) may require more GPU memory
3. **Batch Processing**: For multiple slides, process them sequentially to avoid memory issues

## Support

For issues specific to:
- **hipCIM**: Visit [ROCm-LS/hipCIM](https://github.com/ROCm-LS/hipCIM)
- **MONAI**: Visit [Project MONAI](https://github.com/Project-MONAI/MONAI)
- **ROCm**: Visit [ROCm Documentation](https://rocm.docs.amd.com/)

---

**AMD MONAI Pathology Analysis**
Powered by AMD ROCm • High-Performance Medical Imaging AI