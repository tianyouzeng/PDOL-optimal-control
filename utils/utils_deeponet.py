import torch

# Loss functions for deeponet

def loss_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.flatten() - target.flatten()) ** 2)

def loss_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((1.0 / (pred.shape[1] - 1))**(0.5) \
                      * torch.norm(pred - target, dim=1))

def loss_l2_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.norm(pred - target, dim=1) \
                      / torch.norm(target, dim=1))
