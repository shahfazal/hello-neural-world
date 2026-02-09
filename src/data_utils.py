import torch
import random

def generate_static_curriculum(count=80):
    """
    Generates a fixed set of perfect horizontal and vertical lines.
    Used in 'Discovery' to demonstrate overfitting.
    """
    h_bases = [torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])]
    v_bases = [torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 1.0])]
    
    data = []
    for i in range(count // 2):
        # No noise, just perfect patterns
        data.append((h_bases[i % 2], torch.tensor([1.0, 0.0])))
        data.append((v_bases[i % 2], torch.tensor([0.0, 1.0])))
    return data

def get_dynamic_batch(batch_size=32, noise_level=0.4, device='cpu'):
    """
    Generates a fresh batch of noisy pixels for every training step.
    Used in 'Trainer' to force generalization.
    """
    h_bases = [torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])]
    v_bases = [torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 1.0])]
    
    inputs, targets = [], []
    for _ in range(batch_size // 2):
        # Fresh noise for every single example
        h_noise = (torch.rand(4) * noise_level) - (noise_level / 2)
        v_noise = (torch.rand(4) * noise_level) - (noise_level / 2)
        
        inputs.append(torch.clamp(random.choice(h_bases) + h_noise, 0, 1))
        targets.append(torch.tensor([1.0, 0.0]))
        
        inputs.append(torch.clamp(random.choice(v_bases) + v_noise, 0, 1))
        targets.append(torch.tensor([0.0, 1.0]))
        
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)

def get_stress_test_data():
    """Standard set of ambiguous and perfect cases for qualitative testing."""
    return {
        "Perfect Horizontal": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "Perfect Vertical":   torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "Diagonal Line /":    torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "Solid Gray Block":   torch.tensor([0.5, 0.5, 0.5, 0.5])
    }

def plot_loss_curve(loss_history, title="Training Loss"):
    """Simple matplotlib helper for local logging."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()
