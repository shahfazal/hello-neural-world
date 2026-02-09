import os
import argparse
from huggingface_hub import HfApi, ModelCard, ModelCardData
from pathlib import Path

def publish_model(repo_id, model_path, model_name, metrics=None):
    """
    Publishes a model and its card to the Hugging Face Hub.
    
    Args:
        repo_id (str): The HF repository ID (e.g., 'username/tinynet-horizontal-vertical')
        model_path (str): Path to the .safetensors file
        model_name (str): Human-readable name for the model
        metrics (dict): Optional dictionary of metrics to include in the model card
    """
    api = HfApi()
    
    # 1. Create or verify repo
    print(f"Ensuring repository {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # 2. Upload the model file
    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"{model_name}.safetensors",
        repo_id=repo_id,
        repo_type="model",
    )
    
    # 3. Create a Model Card
    print("Creating model card...")
    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="pytorch",
        tags=["tiny-net", "minimalist", "educational"],
        model_name=model_name,
        metrics=metrics or {}
    )
    
    content = f"""
# {model_name}

This is a **TinyNet** model, part of the "Neural Network Learning Journey."
It is a minimalist 4-3-2 Multi-Layer Perceptron (MLP) designed for educational clarity.

## Model Description
- **Architecture**: 4 Inputs (2x2 grid) → 3 Hidden (Sigmoid) → 2 Outputs (Horizontal/Vertical)
- **Format**: Safetensors

## Training Data
Trained on a static dataset of 4 perfect binary patterns:
- Horizontal (top/bottom)
- Vertical (left/right)

## Intended Use
Educational exploration of backpropagation and overfitting.
"""
    card = ModelCard.from_template(card_data, template_str=content)
    
    # 4. Push the card
    print("Pushing model card to Hub...")
    card.push_to_hub(repo_id)
    
    print(f"\nSuccess! Model published at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a TinyNet model to Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID (e.g. 'faz/tinynet')")
    parser.add_argument("--model", type=str, required=True, help="Path to .safetensors file")
    parser.add_argument("--name", type=str, default="tinynet-discovery", help="Model name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
    else:
        publish_model(args.repo, args.model, args.name)
