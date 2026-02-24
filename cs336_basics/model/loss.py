from collections.abc import Iterable
import torch
import cs336_basics.model.modules
import math
def cross_entropy(logits:torch.Tensor, label: torch.Tensor):
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_prob = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))

    label = label.unsqueeze(-1)

    loss = log_prob.gather(-1, label).squeeze(-1)
    return torch.mean(-loss, -1, keepdim=False)

def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int
)->float:
    if Tw > 0 and t < Tw:
        return (t / Tw )* alpha_max 
    if t > Tc:
        return alpha_min
    progress = (t - Tw) / (Tc - Tw)
    return alpha_min + 1/2 * (1 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)

@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
)->None:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef) 