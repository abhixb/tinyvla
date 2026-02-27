#!/bin/bash
# Quick test script for Qwen2.5-VL-2B model
# Expected VRAM: 4-5 GB

set -e

echo "=================================================="
echo "Testing Qwen2.5-VL-2B with Memory Optimizations"
echo "=================================================="
echo ""
echo "Model: Qwen2.5-VL-2B-Instruct (2B params)"
echo "Optimizations:"
echo "  - 4-bit quantization"
echo "  - LoRA (r=8)"
echo "  - 8-bit optimizer"
echo "  - Gradient checkpointing"
echo ""
echo "Expected VRAM: 4-5 GB"
echo "=================================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import bitsandbytes" 2>/dev/null || {
    echo "ERROR: bitsandbytes not installed"
    echo "Run: uv pip install -e '.[lora]'"
    exit 1
}
echo "✓ bitsandbytes installed"

python -c "import peft" 2>/dev/null || {
    echo "ERROR: peft not installed"
    echo "Run: uv pip install -e '.[lora]'"
    exit 1
}
echo "✓ peft installed"

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: CUDA not available"
    exit 1
}
echo "✓ CUDA available"
echo ""

# Run 10-step test
echo "Running 10-step test..."
echo "Monitor GPU in another terminal: watch -n 1 nvidia-smi"
echo ""

python tinyvla_memory_optimized.py \
    --config config_2gb_test.yaml \
    --max_steps 10 \
    --output_dir ./test_2b_quick

echo ""
echo "=================================================="
echo "Test complete!"
echo ""
echo "Next steps:"
echo "  1. Check peak VRAM was < 6 GB"
echo "  2. If successful, run full training:"
echo "     python tinyvla_memory_optimized.py --config config_2gb_test.yaml"
echo "  3. Then try 3B model:"
echo "     python tinyvla_memory_optimized.py --config config_12gb.yaml --max_steps 10"
echo "=================================================="
