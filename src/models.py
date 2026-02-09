import torch
import torch.nn as nn

class TinyNet(nn.Module):
    """
    TinyNet: A 4-3-2 Multi-Layer Perceptron (MLP).
    Designed for instructional purposes to visualize the 'Neural Nudge'.
    """
    def __init__(self, hidden_activation='sigmoid'):
        super(TinyNet, self).__init__()
        self.layer1 = nn.Linear(4, 3) 
        self.layer2 = nn.Linear(3, 2)
        
        if hidden_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif hidden_activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation. Use 'sigmoid' or 'relu'.")
            
        # Output layer usually stays sigmoid for probability interpretation in this project
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # Hidden layer
        x = self.activation(self.layer1(x))
        # Output layer
        x = self.output_activation(self.layer2(x))
        return x

def get_device():
    """Returns the best available device (MPS for Mac, CUDA for Nvidia, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
