import torch
import torch.nn as nn

def project_to_simplex(v):
    """
    Projects vectors onto the probability simplex (sum to 1, non-negative) 
    using a closed-form sorted-threshold approach.
    """
    shape = v.shape
    if len(shape) > 2:
        v = v.view(shape[0], -1)
        
    n_features = v.shape[1]
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1.0
    ind = torch.arange(1, n_features + 1, device=v.device).float()
    cond = u - cssv / ind > 0
    
    # Find the active coordinate indices boundaries safely
    idx = torch.sum(cond.long(), dim=1, keepdim=True) - 1
    rho = torch.gather(cssv, 1, idx)
    theta = rho / torch.gather(ind.unsqueeze(0).repeat(v.shape[0], 1), 1, idx)
    
    return torch.clamp(v - theta, min=0.0).view(shape)

class LocalDictionaryLoss(torch.nn.Module):
    def __init__(self, penalty):
        super(LocalDictionaryLoss, self).__init__()
        self.penalty = penalty

    def forward(self, A, y, x):
        return self.forward_detailed(A, y, x)[2]

    def forward_detailed(self, A, y, x):
        weight = (y.unsqueeze(1) - A.unsqueeze(0)).pow(2).sum(dim=2)
        a = 0.5 * (y - x @ A).pow(2).sum(dim=1).mean()
        b = (weight * x).sum(dim=1).mean()
        return a, b, a + b * self.penalty