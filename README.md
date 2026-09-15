## SLAI app platform onboarding (demomonai)

This repo is wired for `slai-app-platform` (new `Dockerfile`,
`.github/workflows/deploy-prod.yml`, `deploy/slai-app-prod/demomonai/`).
**Important limitation, by design:** `hipCIM` (AMD's ROCm port of cuCIM) is
ROCm-only and isn't in `requirements.txt` — it's installed separately per
`INSTALLATION.md` on a real Instinct GPU host. `slai-app-prod` is CPU-only
hosting, so it's not installed in the container image at all.

`chatbot_monai_medical.py` was patched so this is a soft limitation, not a
hard crash: the `from cucim import CuImage` import is now wrapped in a
try/except (`CUCIM_AVAILABLE` flag). On this CPU deployment, the app starts
normally and serves the UI, `/api`, and the chatbot's help/greeting replies —
but `/preview-svs`, `/svs-info`, and the "analyze slide" chat command return
a clear 503 / explanatory message instead of the real whole-slide-image
tumor-detection result, since that genuinely requires a ROCm Instinct GPU.
Nothing about this fix changes behavior when hipCIM *is* installed (e.g. a
real GPU deployment elsewhere) — the try/except only affects the case where
it's missing.

Other fixes: `DenseNet121(pretrained=True)`'s weight cache now points at
`$TORCH_HOME=/tmp/torch-cache` (root filesystem is read-only at runtime).

To onboard: set up GitHub Actions secrets/variables on this repo per
`skills/slai-app-creator/SKILL.md` §1c (`SLAI_APP_DEV_PR_TOKEN`,
`HARBOR_USERNAME`, `HARBOR_PASSWORD`; variables `APP_ID=demomonai`,
`IMAGE_NAME=demomonai`, `BUILD_CONTEXT=.`), then Actions → Deploy prod → Run
workflow.

---

# 🔬 AMD MONAI Pathology Analysis

A FastAPI-based web application for automated pathology tumor detection using MONAI's DenseNet121 architecture. This AMD ROCm-accelerated application analyzes whole slide images (WSI) to detect tumor regions in pathological samples.

## 🏥 Features

- **AMD ROCm Accelerated**: Optimized for AMD GPUs with ROCm support
- **AI-Powered Analysis**: Uses MONAI's DenseNet121 for medical image classification
- **hipCIM Integration**: AMD's ROCm port of cuCIM for efficient whole slide image processing
- **Interactive Web UI**: Modern AMD-branded interface with real-time analysis
- **SVS File Support**: Processes whole slide image formats (.svs, .ndpi, .scn)
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Detailed Results**: Provides probability scores, confidence levels, and analysis metrics

## 📋 Prerequisites

- **AMD GPU**: ROCm-compatible AMD GPU (MI200/MI300 series recommended)
- **ROCm 6.0+**: AMD's ROCm platform installed and configured
- **Python 3.10+**: Modern Python environment
- **Ubuntu 22.04**: Or compatible Linux distribution

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YazhRajeshAmd/rocm_monai_pathology.git
cd rocm_monai_pathology

# Install requirements
./setup.sh

### 2. Run the Application

```bash
# Start the server
uvicorn chatbot_monai_medical:app --host 0.0.0.0 --port 8000 --reload

# Access the web interface
# Open http://localhost:8000 in your browser
```

## 📚 Detailed Installation

For comprehensive installation instructions, including system dependencies, ROCm setup, and troubleshooting, see:

**📖 [INSTALLATION.md](INSTALLATION.md)**

## 📦 Requirements
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

## 🔧 Setup

1. **Clone or download the project files**
2. **Place your pathology slide files** in the `data/` directory
   - Supported format: `.svs` (Aperio ScanScope Virtual Slide)
   - Ensure you have a file named `sample_wsi.svs` or modify the code to use your file

## 🚀 Running the Application

### Start the Server

```bash
# Navigate to project directory
cd /rocm_monai_pathology

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
<img width="1383" height="1015" alt="image" src="https://github.com/user-attachments/assets/defd0489-2b1b-406e-af80-37d584698f23" />



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
