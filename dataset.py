import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.datasets import MNIST

class UnitNormFilteredMNIST(Dataset):
    def __init__(self, target_digits=[0, 3, 7], subsample_size=5000, seed=42):
        super().__init__()
        
        mnist_data = MNIST(root='./data', train=True, download=True)
        x_raw = mnist_data.data.numpy().reshape(60000, 784).astype(np.float32) / 255.0
        y_raw = mnist_data.targets.numpy().flatten()
        
        mask = np.isin(y_raw, target_digits)
        x_filtered = x_raw[mask]
        y_filtered = y_raw[mask]
        
        norms = np.linalg.norm(x_filtered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        x_filtered = x_filtered / norms
        
        lookup = np.arange(int(y_filtered.max() + 1))
        lookup[target_digits] = np.arange(len(target_digits))
        y_mapped = lookup[y_filtered]
        
        # Track indices explicitly
        if subsample_size and subsample_size < len(x_filtered):
            np.random.seed(seed)
            # Save the chosen index map array as an attribute
            self.chosen_indices = np.random.permutation(len(x_filtered))[:subsample_size]
            x_filtered = x_filtered[self.chosen_indices]
            y_mapped = y_mapped[self.chosen_indices]
        else:
            self.chosen_indices = np.arange(len(x_filtered))
            
        self.X = torch.tensor(x_filtered, dtype=torch.float32)
        self.y = torch.tensor(y_mapped, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]