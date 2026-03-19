# VibeServer — OpenAI-compatible TTS server for VibeVoice-7B
#
# Target hardware: ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell, ARM64/aarch64)
# Base image:      nvcr.io/nvidia/pytorch:26.02-py3  (CUDA 12.8, PyTorch 2.6, aarch64)
#
# Build:
#   docker buildx build --platform linux/arm64 -t vibeserver:latest .
#
# Run (example):
#   docker run --gpus all -p 8000:8000 \
#     -v /data/hf-cache:/data/hf-cache \
#     -e HF_HOME=/data/hf-cache \
#     -e HF_TOKEN=hf_... \
#     vibeserver:latest

FROM nvcr.io/nvidia/pytorch:26.02-py3

# ---------------------------------------------------------------------------- #
#  System dependencies
# ---------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- #
#  Python dependencies
# ---------------------------------------------------------------------------- #
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the vibevoice inference library from the community fork.
# The official Microsoft repo removed TTS code in Sept 2025; this MIT-licensed
# community fork preserves it.
RUN pip install --no-cache-dir \
    "git+https://github.com/vibevoice-community/VibeVoice.git"

# ---------------------------------------------------------------------------- #
#  Application code + bundled voice presets
# ---------------------------------------------------------------------------- #
COPY app/ /app/

# ---------------------------------------------------------------------------- #
#  Runtime configuration
# ---------------------------------------------------------------------------- #
# Model weights are NOT baked in — they are downloaded on first startup into the
# HuggingFace cache directory.  Set HF_HOME at runtime and mount a persistent
# volume at that path to avoid re-downloading on every container start.
#
# Cache path precedence (standard HuggingFace behaviour):
#   HF_HUB_CACHE  (most specific, overrides everything)
#   $HF_HOME/hub  (if HF_HOME is set)
#   ~/.cache/huggingface/hub  (default fallback)
#
# Recommended: set HF_HOME (e.g. -e HF_HOME=/data/hf-cache) and mount that path.

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# flash_attention_2 is preferred; model.py falls back to sdpa automatically
# if it is unavailable on this CUDA build.
ENTRYPOINT ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
