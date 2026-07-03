import torch
import torch.nn as nn
from loss import project_to_simplex

class KDS(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        penalty,
        accelerate=True,
        train_step=True,
        W=None,
        step=None,
    ):
        super(KDS, self).__init__()

        # Register hyperparameters as buffers safely
        self.register_buffer("num_layers", torch.tensor(int(num_layers)))
        self.register_buffer("input_size", torch.tensor(int(input_size)))
        self.register_buffer("hidden_size", torch.tensor(int(hidden_size)))
        self.register_buffer("penalty", torch.tensor(float(penalty)))
        self.register_buffer("accelerate", torch.tensor(bool(accelerate)))

        # Standardize parameter dimensions: W has shape [input_size, hidden_size] -> (784, m)
        if W is None:
            W = torch.zeros(self.hidden_size, self.input_size)
            
        self.register_parameter("W", torch.nn.Parameter(W))
        
        # Safe lipschitz step constant initialization avoiding all-zero SVD infinite faults
        if step is None:
            # SVD expects [M, N]. Using safe fallback estimation if matrix is singular
            try:
                U, S, V = torch.svd(self.W.data)
                max_singular_val = S[0].item()
                step = torch.tensor(max_singular_val ** -2) if max_singular_val > 1e-5 else torch.tensor(1e-3)
            except Exception:
                step = torch.tensor(1e-3)
                
        if train_step:
            self.register_parameter("step", torch.nn.Parameter(step))
        else:
            self.register_buffer("step", step)
        
    def encode_accelerated(self, y):
        x_tmp = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        x_old = torch.zeros(y.shape[0], self.hidden_size, device=y.device)
        
        # Distance weight calculations: mapping spatial proximity rules
        # With shape matching: y is [B, 784], W is [784, m]
        weight = (
            y.square().sum(dim=1, keepdim=True)
            + self.W.T.square().sum(dim=0, keepdim=True)
            - 2 * y @ self.W.T
        )
        
        for layer in range(self.num_layers.item()):
            grad = (x_tmp @ self.W - y) @ self.W.T
            grad = grad + weight * self.penalty
            x_new = self.activate(x_tmp - grad * self.step)
            x_old, x_tmp = x_new, x_new + layer / (layer + 3) * (x_new - x_old)
                
        return x_new

    def forward(self, y):
        """
        The formal forward pass expected by stream_engine.py.
        Returns:
            y_hat: The high-dimensional projection onto the current simplex hull [Batch, 784]
            x: The optimal sparse latent codes [Batch, Hidden_Size]
        """
        # 1. Calculate the optimal sparse coefficients mapping
        x = self.encode_accelerated(y)
        
        # 2. Reconstruct the high-dimensional image estimation vector (y_hat = x @ W.T)
        # x is [Batch, m], self.W.T is [m, 784] -> output y_hat is [Batch, 784]
        y_hat = x @ self.W
        
        return y_hat, x
    
    def activate(self, x):
        m, n = x.shape
        cnt_m = torch.arange(m, device=x.device)
        cnt_n = torch.arange(n, device=x.device)
        u = x.sort(dim=1, descending=True).values
        v = (u.cumsum(dim=1) - 1) / (cnt_n + 1)
        w = v[cnt_m, (u > v).sum(dim=1) - 1]
        return (x - w.view(m, 1)).relu()