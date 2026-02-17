#!/bin/bash
# TinyVLA H100 Setup - Optimized for NVIDIA H100 GPUs
set -e

echo "TinyVLA H100 Setup"
echo "=================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install
echo "Creating environment..."
uv venv .venv --python 3.11 --clear
source .venv/bin/activate

# Install PyTorch with CUDA 12.4 (optimal for H100)
echo "Installing PyTorch (CUDA 12.4)..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (H100 has great support)
echo "Installing Flash Attention..."
if [ -z "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME environment variable is not set."
    echo "Please install the CUDA Toolkit and set CUDA_HOME to your CUDA installation directory."
    echo "You can download the CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
uv pip install setuptools wheel
uv pip install flash-attn --no-build-isolation
uv pip install -e '.[lora]'

# Install TinyVLA with eval + LeRobot dependencies
echo "Installing TinyVLA..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[eval,lerobot]"

echo ""
echo "Setup complete! (H100 optimized)"
echo ""
echo "Quick test:"
echo "  python tinyvla.py --test"
echo ""
echo "Full training (H100 80GB - no LoRA needed):"
echo "  python tinyvla.py --epochs 24 --batch-size 16"
echo ""
echo "Multi-GPU training:"
echo "  accelerate launch --num_processes 4 tinyvla.py --epochs 24 --batch-size 16"
echo ""
