#!/bin/bash
# TinyVLA Cloud Setup - Fire and Forget Training
set -e

echo "TinyVLA Setup"
echo "============="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install
echo "Creating environment..."
uv venv .venv
source .venv/bin/activate

# Install PyTorch first (CUDA 12.1)
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install TinyVLA with eval dependencies
echo "Installing TinyVLA..."
uv pip install -e ".[eval]"

# Install LIBERO
if [ ! -d "/tmp/LIBERO" ]; then
    echo "Installing LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
    pip install -e /tmp/LIBERO
fi

echo ""
echo "Setup complete!"
echo ""
echo "Quick test:"
echo "  python tinyvla.py --test --use-lora"
echo ""
echo "Full training (fire-and-forget):"
echo "  python tinyvla.py --epochs 32 --batch-size 8 --eval-every 8 --eval-episodes 20"
echo ""
echo "With LoRA (if OOM on <80GB GPU):"
echo "  python tinyvla.py --epochs 32 --batch-size 8 --eval-every 8 --eval-episodes 20 --use-lora"
echo ""
