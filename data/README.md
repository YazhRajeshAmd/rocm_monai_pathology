# 🔬 MONAI Pathology Tumor Detection Chatbot

A FastAPI-based chatbot for automated pathology tumor detection using MONAI's DenseNet121 architecture. This application analyzes whole slide images (WSI) to detect tumor regions in pathological samples.

## 🏥 Features

- **AI-Powered Analysis**: Uses MONAI's DenseNet121 for medical image classification
- **Whole Slide Image Support**: Processes .svs files using cuCIM for efficient WSI handling
- **ROCm/CUDA Compatible**: Supports both AMD (ROCm) and NVIDIA (CUDA) GPUs
- **RESTful API**: FastAPI-based web service with interactive documentation
- **Detailed Results**: Provides probability scores, confidence levels, and analysis metrics

## 📋 Prerequisites

- Python 3.8+
- AMD GPU with ROCm support OR NVIDIA GPU with CUDA support
- Linux environment (tested on Ubuntu)

## 🚀 Installation Steps

### 1. System Dependencies

For ROCm (AMD GPUs):
```bash
# Install ROCm (if not already installed)
# Follow official ROCm installation guide for your system
```

### 2. Python Package Requirements

Install the required Python packages:

```bash
# Core web framework
pip3 install fastapi uvicorn

# Medical imaging libraries
pip3 install monai
pip3 install cucim

# Deep learning frameworks
pip3 install torch torchvision

# Scientific computing
pip3 install numpy

# For ROCm users, you might need:
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2

# For CUDA users, ensure PyTorch CUDA version matches your system:
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Alternative: Requirements File Installation

Create a `requirements.txt` file:
```txt
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
monai>=1.0.0
cucim
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
```

Then install:
```bash
pip3 install -r requirements.txt
```

### 4. Verify Installation

Test MONAI installation:
```bash
python3 -c "import monai; print('MONAI version:', monai.__version__)"
```

Test GPU availability:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 📁 Project Structure

```
pathology-chatbot/
├── chatbot_monai_medical.py    # Main application file
├── data/                       # Directory for pathology slides
│   └── sample_wsi.svs         # Sample whole slide image
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Install hipCIM via AMD PyPI
[Optional step] Follow these if you want to install hipCIM inside a docker

 docker pull rocm/dev-ubuntu-22.04
 docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true   \
      --shm-size=128GB --network=host --device=/dev/kfd     \
      --device=/dev/dri --group-add video -it               \
      -v $HOME:$HOME  --name ${LOGNAME}_rocm                \
  rocm/dev-ubuntu-22.04:6.4.1-complete
Install required system dependencies

 sudo apt update
 sudo apt install -y lsb-release software-properties-common libopenslide0 python3.10-venv rocjpeg
 sudo apt install -y rocthrust-dev hipcub hipblas \
            hipblas-dev hipfft hipsparse \
            hiprand rocsolver rocrand-dev
 pip install --upgrade pip
Create a python virtual environment

 python3 -m venv hipcim_build
 source hipcim_build/bin/activate
 
     # Install hipCIM
 pip install amd-hipcim --extra-index-url=https://pypi.amd.com/simple
Checkout jupyter notebooks

pip install notebook
git clone --depth 1 git@github.com:ROCm-LS/hipCIM.git hipcim-notebooks \
   && cd hipcim-notebooks \
   && git filter-branch --prune-empty --subdirectory-filter notebooks HEAD
git lfs pull

## 🔧 Setup

1. **Clone or download the project files**
2. **Place your pathology slide files** in the `data/` directory
   - Supported format: `.svs` (Aperio ScanScope Virtual Slide)
   - Ensure you have a file named `sample_wsi.svs` or modify the code to use your file

## 🚀 Running the Application

### Start the Server

```bash
# Navigate to project directory
cd /path/to/your/project

# Start the FastAPI server
uvicorn chatbot_monai_medical:app --host 0.0.0.0 --port 8000 --reload
```

The server will start and display:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
Loading MONAI DenseNet121 for pathology tumor detection...
Using device: cuda
MONAI DenseNet121 model loaded successfully for pathology tumor detection
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **API Schema**: http://localhost:8000/redoc

## 📡 API Usage

### Chat Endpoint

**POST** `/chat`

**Request Body**:
```json
{
    "user": "analyze slide"
}
```

**Response**:
```json
{
    "reply": "🔬 **Pathology Analysis Results:**\n📊 **Prediction:** Normal Tissue\n📈 **Tumor Probability:** 0.123\n📉 **Normal Probability:** 0.877\n🎯 **Confidence Level:** High (0.877)\n📍 **Region Analyzed:** 512x512 patch at location (0,0)\n⚙️ **Model:** MONAI DenseNet121 for Pathology",
    "history": [...]
}
```

### Available Commands

- `"analyze slide"` - Perform tumor detection analysis
- `"tumor detection"` - Alternative analysis command
- `"help"` - Show available commands
- Any other text - General chatbot response

### Example Usage

```bash
# Analyze pathology slide
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"user": "analyze slide"}'

# Get help
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"user": "help"}'

# Check API status
curl http://localhost:8000/
```

## 🔬 Technical Details

### Model Architecture
- **Base Model**: MONAI DenseNet121
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Normal vs Tumor)
- **Preprocessing**: MONAI transforms with intensity normalization

### Image Processing Pipeline
1. Load WSI using cuCIM
2. Extract 512x512 region at specified location
3. Convert from HWC to CHW format
4. Resize to 224x224
5. Apply intensity normalization
6. Run inference with DenseNet121
7. Generate probability scores and predictions

### GPU Support
- **ROCm (AMD)**: Automatic detection and usage
- **CUDA (NVIDIA)**: Automatic detection and usage  
- **CPU Fallback**: If no GPU available

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error for MONAI transforms**:
   ```bash
   # Ensure MONAI is properly installed
   pip3 install --upgrade monai
   ```

2. **CUDA/ROCm not detected**:
   ```bash
   # Verify GPU drivers and runtime
   nvidia-smi  # For NVIDIA
   rocm-smi    # For AMD
   ```

3. **cuCIM import errors**:
   ```bash
   # Install cuCIM from conda-forge if pip fails
   conda install -c conda-forge cucim
   ```

4. **File not found errors**:
   - Ensure `data/sample_wsi.svs` exists
   - Check file permissions
   - Verify file format (.svs)

### Performance Optimization

- **Large WSI files**: Adjust region size and location for analysis
- **Memory usage**: Reduce batch size or image resolution if needed
- **GPU memory**: Monitor VRAM usage with large slide files

## 📊 Model Performance

This is a demonstration model using transfer learning. For production use:
- Train on domain-specific pathology datasets
- Implement proper validation and testing
- Add confidence thresholding
- Consider ensemble methods

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical data regulations when using with real pathology data.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval.

---

**Happy Analyzing! 🔬🏥✨**