FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Platform deployment is CPU-only: install from requirements.txt as-is.
# hipCIM/amd-hipcim (ROCm-only, from pypi.amd.com) is intentionally NOT
# installed here — chatbot_monai_medical.py now guards that import so the
# app still starts and serves everything except whole-slide-image features.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DenseNet121(pretrained=True) downloads ImageNet weights on first start and
# needs a writable torch cache — root filesystem is read-only at runtime.
ENV TORCH_HOME=/tmp/torch-cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "chatbot_monai_medical:app", "--host", "0.0.0.0", "--port", "8000"]
