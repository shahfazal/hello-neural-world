# Hello Neural World

A minimal neural network learning project with a 4→3→2 architecture. Learn backpropagation, overfitting, and generalization through hands-on experimentation.

## What's Inside

- **2 Jupyter Notebooks**: Step-by-step tutorials from basics to generalization
- **Python Package**: Reusable training engine, models, and data utilities
- **3 Trained Models**: Published on Hugging Face with full documentation

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook runbooks/tinynet_discovery.ipynb
```

**Notebooks:**
1. [tinynet_discovery.ipynb](runbooks/tinynet_discovery.ipynb) - Learn how backpropagation works
2. [tinynet_trainer.ipynb](runbooks/tinynet_trainer.ipynb) - Fix overfitting with noise and regularization

## Models

Three trained models available on Hugging Face:

- [tinynet-sigmoid-baseline](https://huggingface.co/[username]/tinynet-sigmoid-baseline) - Discovery model (overfits)
- [tinynet-relu-v1](https://huggingface.co/[username]/tinynet-relu-v1) - ReLU upgrade
- [tinynet-v3-regularized](https://huggingface.co/[username]/tinynet-v3-regularized) - Final model with weight decay

## Blog Post

Read the full learning journey: [Coming soon on Medium]

## Tech Stack

PyTorch · Safetensors · Weights & Biases · Jupyter

## License

MIT