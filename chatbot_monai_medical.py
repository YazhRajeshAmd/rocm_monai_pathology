# chatbot_monai_pathology_tumor_detection.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from cucim import CuImage
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
import shutil
from PIL import Image
import io
import base64
from monai.transforms import (
    Compose, LoadImage, ScaleIntensity, Resize, ToTensor, EnsureChannelFirst, Lambda,
    RandFlip, RandRotate, NormalizeIntensity
)
from monai.networks.nets import DenseNet121
from monai.apps import download_and_extract
from monai.apps.mmars import download_mmar, MODEL_DESC

# -------------------------
# Setup FastAPI app with static file serving
# -------------------------
app = FastAPI(title="AMD MONAI Pathology Analysis", description="AI-Powered Tumor Detection")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

chat_history = []

class ChatRequest(BaseModel):
    user: str

class AnalysisResult(BaseModel):
    reply: str
    history: list
    image_path: str = None
    analysis_data: dict = None

# Create directories if they don't exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -------------------------
# Serve the main UI
# -------------------------
@app.get("/")
async def serve_ui():
    """Serve the main AMD-branded UI"""
    return FileResponse('static/index.html')

# -------------------------
# API Info endpoint
# -------------------------
@app.get("/api")
async def api_info():
    return {
        "message": "🏥 AMD MONAI Pathology Tumor Detection API",
        "description": "AI-powered pathology analysis using MONAI DenseNet121",
        "version": "1.0.0",
        "framework": "MONAI + FastAPI",
        "model": "DenseNet121",
        "gpu_support": "AMD ROCm / NVIDIA CUDA",
        "endpoints": {
            "/": "GET - Main UI Interface",
            "/chat": "POST - Chat with AI pathologist",
            "/api": "GET - API information",
            "/docs": "GET - API documentation"
        }
    }

# -------------------------
# Load pretrained model for Pathology Tumor Detection
# -------------------------
print("Loading MONAI DenseNet121 for pathology tumor detection...")

# Check if ROCm/HIP is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use MONAI's DenseNet121 which is optimized for medical imaging
# This is commonly used for pathology classification tasks
model = DenseNet121(
    spatial_dims=2,      # 2D images (pathology slides)
    in_channels=3,       # RGB channels
    out_channels=2,      # Binary classification: Normal vs Tumor
    pretrained=True      # Use ImageNet pretrained weights as starting point
)

model = model.to(device)
model.eval()

print("MONAI DenseNet121 model loaded successfully for pathology tumor detection")

# -------------------------
# Pathology-specific preprocessing pipeline
# -------------------------
# Optimized preprocessing for pathology tumor detection
preprocess = Compose([
    # Convert numpy array to proper format first
    Lambda(lambda x: np.transpose(x, (2, 0, 1)) if x.ndim == 3 and x.shape[2] <= 4 else x),  # HWC to CHW
    Lambda(lambda x: x[:3] if x.shape[0] >= 3 else x),  # Take only first 3 channels if more than 3
    Lambda(lambda x: np.stack([x[0], x[0], x[0]]) if x.shape[0] == 1 else x),  # Convert grayscale to RGB if needed
    Resize((224, 224)),         # Standard input size for DenseNet121
    ScaleIntensity(minv=0.0, maxv=1.0),  # Normalize to [0,1]
    NormalizeIntensity(),       # MONAI intensity normalization
    ToTensor(),                 # Convert to PyTorch tensor
])

# Additional preprocessing for data augmentation (can be used during training)
augment_preprocess = Compose([
    Lambda(lambda x: np.transpose(x, (2, 0, 1)) if x.ndim == 3 and x.shape[2] <= 4 else x),
    Lambda(lambda x: x[:3] if x.shape[0] >= 3 else x),
    Lambda(lambda x: np.stack([x[0], x[0], x[0]]) if x.shape[0] == 1 else x),
    RandFlip(prob=0.5, spatial_axis=1),    # Random horizontal flip (axis 1 in CHW format)
    RandFlip(prob=0.5, spatial_axis=2),    # Random vertical flip (axis 2 in CHW format)  
    RandRotate(prob=0.5, range_x=0.1),     # Small random rotations
    Resize((224, 224)),
    ScaleIntensity(minv=0.0, maxv=1.0),
    NormalizeIntensity(),
    ToTensor(),
])

# -------------------------
# Enhanced Chat endpoint for Pathology Tumor Detection
# -------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = req.user
    chat_history.append({"role": "user", "content": user_msg})

    if "analyze slide" in user_msg.lower() or "tumor detection" in user_msg.lower():
        try:
            # Load slide with cuCIM for efficient whole slide image processing
            img = CuImage("data/sample_wsi.svs")
            
            # Extract a region for analysis (you can modify location and size)
            # For real applications, you might analyze multiple regions
            region = img.read_region(location=(0, 0), size=(512, 512), level=0)

            # Convert to numpy array
            arr = np.asarray(region)   # numpy HWC format
            print(f"Original image shape: {arr.shape}")
            
            # Preprocess using pathology-optimized pipeline
            patch = preprocess(arr)   # [C,H,W] tensor
            print(f"Patch shape after preprocessing: {patch.shape}")
            patch = patch.unsqueeze(0).to(device)  # add batch dimension and move to device

            # Inference with MONAI DenseNet121
            with torch.no_grad():
                logits = model(patch)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                confidence = torch.max(torch.softmax(logits, dim=1)).item()

            normal_prob = float(probabilities[0][0])
            tumor_prob = float(probabilities[0][1])
            
            # Determine prediction
            prediction = "Tumor Detected" if tumor_prob > normal_prob else "Normal Tissue"
            confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"

            bot_msg = (
                f"🔬 **Pathology Analysis Results:**\n"
                f"📊 **Prediction:** {prediction}\n"
                f"📈 **Tumor Probability:** {tumor_prob:.3f}\n"
                f"📉 **Normal Probability:** {normal_prob:.3f}\n"
                f"🎯 **Confidence Level:** {confidence_level} ({confidence:.3f})\n"
                f"📍 **Region Analyzed:** 512x512 patch at location (0,0)\n"
                f"⚙️ **Model:** MONAI DenseNet121 for Pathology"
            )
            
        except Exception as e:
            bot_msg = f"Error during pathology analysis: {str(e)}. Please ensure the slide file exists at data/sample_wsi.svs"
            
    elif "help" in user_msg.lower() or "commands" in user_msg.lower():
        bot_msg = (
            "🏥 **Pathology Tumor Detection Chatbot**\n\n"
            "Available commands:\n"
            "• 'analyze slide' - Analyze pathology slide for tumor detection\n"
            "• 'tumor detection' - Same as analyze slide\n"
            "• 'help' - Show this help message\n\n"
            "🔬 This chatbot uses MONAI's DenseNet121 for medical pathology analysis.\n"
            "📁 Place your .svs slide files in the data/ directory."
        )
    else:
        bot_msg = (
            "🏥 I'm a specialized pathology tumor detection chatbot powered by MONAI!\n"
            "Try: 'analyze slide', 'tumor detection', or 'help' for available commands."
        )

    chat_history.append({"role": "assistant", "content": bot_msg})
    return {"reply": bot_msg, "history": chat_history}
