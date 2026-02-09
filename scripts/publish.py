import os
import argparse
from huggingface_hub import HfApi, ModelCard, ModelCardData
from pathlib import Path

def publish_model(repo_id, model_path, model_name, metrics=None, variant="sigmoid"):
    """
    Publishes a model and its card to the Hugging Face Hub.

    Args:
        repo_id (str): The HF repository ID (e.g., 'username/tinynet-sigmoid-baseline')
        model_path (str): Path to the .safetensors file
        model_name (str): Human-readable name for the model
        metrics (dict): Optional dictionary of metrics to include in the model card
        variant (str): Model variant - 'sigmoid', 'relu-v1', or 'relu-v2'
    """
    api = HfApi()

    # 1. Create or verify repo
    print(f"Ensuring repository {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # 2. Upload the model file
    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.safetensors",
        repo_id=repo_id,
        repo_type="model",
    )

    # 3. Create variant-specific model card content
    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="pytorch",
        tags=["tiny-model", "image-classification", "educational", variant],
        model_name=model_name,
    )

    descriptions = {
        "sigmoid": """
## About This Model

**TinyNet Sigmoid Baseline** - The first generalization attempt from the Trainer notebook.

**Key characteristics:**
- Sigmoid activation on both hidden and output layers
- Trained on noisy data (noise_level=0.3) to prevent overfitting
- Uses SGD optimizer with learning rate 0.1
- 200 training epochs

**Performance:**
- Final loss: ~0.251
- Shows improved generalization over Discovery model
- Still exhibits saturation issues typical of Sigmoid activations

**From the blog post:** This model demonstrates that adding noise helps, but Sigmoid still struggles with gradient flow.""",

        "relu-v1": """
## About This Model

**TinyNet ReLU v1** - Second iteration with ReLU activation upgrade.

**Key improvements:**
- ReLU activation on hidden layer (better gradient flow)
- Sigmoid only on output layer (for probability interpretation)
- Same noise strategy as Sigmoid baseline
- Uses SGD optimizer with learning rate 0.1
- 200 training epochs

**Performance:**
- Final loss: ~0.224 (11% better than Sigmoid)
- Faster convergence
- Better confidence on clear patterns
- Less saturation issues

**From the blog post:** ReLU activation shows clear advantages but still needs refinement for production use.""",

        "relu-v2": """
## About This Model

**TinyNet ReLU v2 Regularized** - Final production-ready model with multiple improvements.

**Key improvements:**
- ReLU on hidden layer, Sigmoid on output
- **Mixed batches**: 20% clean + 80% noisy data (learns confidence + generalization)
- **Adam optimizer**: Adaptive learning rate (lr=0.01)
- **Weight decay (L2 regularization)**: 0.01 to prevent overfitting
- **Extended training**: 300 epochs (regularization allows longer training)

**Performance:**
- Final loss: ~0.057 (77% better than Sigmoid baseline!)
- High confidence (>95%) on clear patterns
- Appropriate uncertainty (~50%) on ambiguous cases
- Production-ready generalization

**From the blog post:** This model demonstrates real ML iteration - each weakness identified, each improvement targeted. The result: a model that knows what it knows."""
    }

    content = f"""
Part of the [Hello Neural World](https://github.com/shahfazal/hello-neural-world) learning project.

{descriptions.get(variant, descriptions["sigmoid"])}

## Architecture

```
Input Layer:  4 neurons (2x2 pixel grid)
             ↓
Hidden Layer: 3 neurons (ReLU or Sigmoid)
             ↓
Output Layer: 2 neurons (Horizontal vs Vertical probabilities)
```

**Total parameters:** 23 (4×3 + 3 bias + 3×2 + 2 bias)

## Training Data

Trained on thousands of noisy examples generated from 4 base patterns:
- Horizontal top: [1,1,0,0]
- Horizontal bottom: [0,0,1,1]
- Vertical left: [1,0,1,0]
- Vertical right: [0,1,0,1]

Each pattern augmented with random noise to force pattern learning instead of memorization.

## Usage

```python
from safetensors.torch import load_file
import torch.nn as nn

# Define the architecture
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.layer1 = nn.Linear(4, 3)
        self.layer2 = nn.Linear(3, 2)
        self.relu = nn.ReLU()  # or nn.Sigmoid() for baseline
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Load weights
model = TinyNet()
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)

# Run inference
import torch
test_input = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # Perfect horizontal
output = model(test_input)
print(f"Horizontal: {{output[0][0]:.2%}}, Vertical: {{output[0][1]:.2%}}")
```

## Intended Use

Educational purposes - demonstrates:
- Backpropagation mechanics
- Effect of activation functions
- Overfitting vs generalization
- Impact of data augmentation (noise)
- Iterative ML development process

## Limitations

- Toy dataset (2×2 grids only)
- Binary classification (horizontal vs vertical)
- Not for production use
- Designed for learning, not performance

## Learn More

- **Notebooks**: [Discovery](https://github.com/shahfazal/hello-neural-world/blob/main/runbooks/tinynet_discovery.ipynb) | [Trainer](https://github.com/shahfazal/hello-neural-world/blob/main/runbooks/tinynet_trainer.ipynb)
- **Blog Post**: [Coming soon on Medium]
- **GitHub**: [hello-neural-world](https://github.com/shahfazal/hello-neural-world)

## License

MIT
"""
    # Create the full model card with YAML metadata
    card = ModelCard(content)
    card.data = card_data

    # 4. Write README.md to the repo
    print("Pushing model card to Hub...")
    api.upload_file(
        path_or_fileobj=str(card).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\n✓ Success! Model published at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a TinyNet model to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID (e.g. 'username/tinynet-sigmoid-baseline')")
    parser.add_argument("--model", type=str, required=True, help="Path to .safetensors file")
    parser.add_argument("--name", type=str, required=True, help="Model display name")
    parser.add_argument("--variant", type=str, choices=["sigmoid", "relu-v1", "relu-v2"], default="sigmoid",
                        help="Model variant (sigmoid, relu-v1, or relu-v2)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(args.model)}")
    else:
        publish_model(args.repo, args.model, args.name, variant=args.variant)
