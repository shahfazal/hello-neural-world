import torch
import torch.nn as nn
from safetensors.torch import save_file
import os

def train_model(model, optimizer, criterion, data_loader_fn, epochs=100, device='cpu', use_wandb=False, model_name="tinynet"):
    """
    Generic training engine that can be used by both Student and Publisher workflows.
    """
    model.to(device)
    loss_history = []
    
    print(f"Starting training for {model_name} on {device}...")
    
    for epoch in range(epochs):
        model.train()
        # Get data for this step
        inputs, targets = data_loader_fn() 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if use_wandb:
            import wandb
            wandb.log({"loss": loss_val, "epoch": epoch})
            
        if epoch % (max(1, epochs // 10)) == 0:
            print(f"Epoch {epoch:03} | Loss: {loss_val:.4f}")
            
    print("Training Complete.")
    return loss_history

def export_model(model, path):
    """Securely saves the model state as a safetensors file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.to('cpu').state_dict()
    save_file(state_dict, path)
    print(f"Model saved to {path} (Safetensors)")
