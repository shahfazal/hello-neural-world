# Publishing Scripts

Helper scripts for publishing TinyNet models to Hugging Face.

## Prerequisites

```bash
pip install huggingface-hub
hf auth login
```

## Quick Start

### Option 1: Publish All Models (Recommended)

```bash
# After running the Trainer notebook export cell
./scripts/publish_all.sh YOUR_HF_USERNAME
```

### Option 2: Publish Individual Models

```bash
# Sigmoid baseline
python scripts/publish.py \
    --repo YOUR_USERNAME/tinynet-sigmoid-baseline \
    --model models/tinynet-sigmoid-baseline.safetensors \
    --name "TinyNet Sigmoid Baseline" \
    --variant sigmoid

# ReLU v1
python scripts/publish.py \
    --repo YOUR_USERNAME/tinynet-relu-v1 \
    --model models/tinynet-relu-v1.safetensors \
    --name "TinyNet ReLU v1" \
    --variant relu-v1

# ReLU v2 Regularized
python scripts/publish.py \
    --repo YOUR_USERNAME/tinynet-relu-v2-regularized \
    --model models/tinynet-relu-v2-regularized.safetensors \
    --name "TinyNet ReLU v2 Regularized" \
    --variant relu-v2
```

## Workflow

1. **Train models**: Run `runbooks/tinynet_trainer.ipynb` to completion
2. **Export models**: Run the final export cell in the notebook (creates `.safetensors` files)
3. **Authenticate**: `huggingface-cli login` (one-time setup)
4. **Publish**: Run `./scripts/publish_all.sh YOUR_USERNAME`
5. **Update README**: Replace `[username]` placeholders with your actual HF username

## What Gets Published

Each model repo will include:
- `model.safetensors` - The trained weights
- `README.md` - Comprehensive model card with:
  - Model description and architecture
  - Training details and performance metrics
  - Usage examples
  - Links to notebooks and blog post
  - Educational context

## Model Repos Created

- `YOUR_USERNAME/tinynet-sigmoid-baseline` - Baseline with Sigmoid
- `YOUR_USERNAME/tinynet-relu-v1` - First ReLU upgrade
- `YOUR_USERNAME/tinynet-relu-v2-regularized` - Production model with regularization
