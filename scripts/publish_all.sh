#!/bin/bash
# Helper script to publish all TinyNet models to Hugging Face

# Check if username is provided
if [ -z "$1" ]; then
    echo "Usage: ./publish_all.sh YOUR_HF_USERNAME"
    echo "Example: ./publish_all.sh shahfazal"
    exit 1
fi

USERNAME=$1

echo "Publishing TinyNet models to Hugging Face..."
echo "Username: $USERNAME"
echo ""

# Check if user is logged in to HF
if ! hf auth whoami &> /dev/null; then
    echo "Error: Not logged in to Hugging Face"
    echo "Please run: hf auth login"
    exit 1
fi

echo "✓ Hugging Face authentication verified"
echo ""

# Publish Sigmoid baseline
echo "📦 Publishing Sigmoid Baseline..."
python scripts/publish.py \
    --repo "$USERNAME/tinynet-sigmoid-baseline" \
    --model models/tinynet-sigmoid-baseline.safetensors \
    --name "TinyNet Sigmoid Baseline" \
    --variant sigmoid

echo ""

# Publish ReLU v1
echo "📦 Publishing ReLU v1..."
python scripts/publish.py \
    --repo "$USERNAME/tinynet-relu-v1" \
    --model models/tinynet-relu-v1.safetensors \
    --name "TinyNet ReLU v1" \
    --variant relu-v1

echo ""

# Publish ReLU v2
echo "📦 Publishing ReLU v2 Regularized..."
python scripts/publish.py \
    --repo "$USERNAME/tinynet-relu-v2-regularized" \
    --model models/tinynet-relu-v2-regularized.safetensors \
    --name "TinyNet ReLU v2 Regularized" \
    --variant relu-v2

echo ""
echo "✅ All models published successfully!"
echo ""
echo "View your models at:"
echo "  • https://huggingface.co/$USERNAME/tinynet-sigmoid-baseline"
echo "  • https://huggingface.co/$USERNAME/tinynet-relu-v1"
echo "  • https://huggingface.co/$USERNAME/tinynet-relu-v2-regularized"
