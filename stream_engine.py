import torch
import numpy as np
from torchvision.datasets import MNIST

class ProductionStreamSimulator:
    def __init__(self, target_digits=[0, 3, 7], total_stream_samples=1000, target_ratio=0.85, seed=42):
        """
        Simulates an incoming data stream interleaving target (in-distribution) data 
        and anomalous/novel class data (out-of-distribution) at a specified ratio.
        """
        self.target_digits = target_digits
        self.total_stream_samples = total_stream_samples
        self.target_ratio = target_ratio
        self.rng = np.random.default_rng(seed)
        
        # Load and flatten raw MNIST vectors
        mnist_data = MNIST(root='./data', train=True, download=True)
        self.X_raw = mnist_data.data.numpy().reshape(60000, 784).astype(np.float32) / 255.0
        self.y_raw = mnist_data.targets.numpy().flatten()

    def generate_interleaved_stream(self, trained_indices_to_exclude=None):
        """
        Assembles and shuffles the interleaved stream pool.
        Ensures absolute data isolation by excluding indices used in baseline dictionary training.
        """
        # Create masks separating targets from anomalous classes
        mask = np.isin(self.y_raw, self.target_digits)
        X_target_pool = self.X_raw[mask]
        y_target_pool = self.y_raw[mask]
        
        # Enforce strict zero data leakage compliance
        if trained_indices_to_exclude is not None:
            elements_to_keep_mask = np.ones(X_target_pool.shape[0], dtype=bool)
            elements_to_keep_mask[trained_indices_to_exclude] = False
            X_target_pool = X_target_pool[elements_to_keep_mask]
            y_target_pool = y_target_pool[elements_to_keep_mask]
            
        X_novel_pool = self.X_raw[~mask]
        y_novel_pool = self.y_raw[~mask]
        
        # Calculate pool budget splits
        num_target = int(self.total_stream_samples * self.target_ratio)
        num_novel = self.total_stream_samples - num_target
        
        # Randomly sample allocations
        tgt_idx = self.rng.choice(len(X_target_pool), size=num_target, replace=False)
        novel_idx = self.rng.choice(len(X_novel_pool), size=num_novel, replace=False)
        
        X_t, y_t = X_target_pool[tgt_idx], y_target_pool[tgt_idx]
        X_n, y_n = X_novel_pool[novel_idx], y_novel_pool[novel_idx]
        
        # Project vectors onto the unit sphere surface
        X_t = X_t / np.linalg.norm(X_t, axis=1, keepdims=True)
        X_n = X_n / np.linalg.norm(X_n, axis=1, keepdims=True)
        
        # Merge streams together
        total_subset = np.concatenate([X_t, X_n], axis=0)
        total_labels = np.concatenate([y_t, y_n])
        in_target = np.concatenate([np.ones(len(X_t)), np.zeros(len(X_n))]).astype(int)
        
        # Uniformly randomize sequence distribution order
        perm = self.rng.permutation(len(total_subset))
        return total_subset[perm], total_labels[perm], in_target[perm]

class IncrementalStreamEngine:
    def __init__(self, model, threshold=1e-4):
        self.model = model
        self.threshold = threshold

    def process_streaming_point(self, streaming_vector, is_in_target=1):
        self.model.eval()
        with torch.no_grad():
            # Ensure tensor has a batch dimension
            if len(streaming_vector.shape) == 1:
                tensor_input = torch.tensor(streaming_vector, dtype=torch.float32).unsqueeze(0)
            else:
                tensor_input = torch.tensor(streaming_vector, dtype=torch.float32)
            
            # Forward pass through unrolled PGD encoder
            # y_hat is the closest projection onto the current convex hull
            y_hat, _ = self.model(tensor_input)
            
            # Compute true Euclidean geometric distance
            recon_error = torch.norm(tensor_input - y_hat, p=2).item()
            
            did_expand = False
            # Check if the point breaches the convex hull boundary threshold
            if recon_error > self.threshold:
                did_expand = True
                
                # Append the raw out-of-hull vector as a new dictionary atom row
                self.model.W.data = torch.cat([self.model.W.data, tensor_input], dim=0)

                # Re-normalize each atom (row) to keep them on the unit sphere
                self.model.W.data /= torch.norm(self.model.W.data, dim=1, keepdim=True)
                self.model.hidden_size += 1

            # Return projection coordinates along with metrics
            return did_expand, recon_error, y_hat.squeeze(0).numpy()