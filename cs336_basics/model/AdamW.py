from turtle import window_height
from typing import Optional
from collections.abc import Callable, Iterable
import torch
import torch.nn

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid learning rate: {eps}")
        if weight_decay <= 0.0:
            raise ValueError(f"Invalid learning rate: {weight_decay}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 <= 1.0):
            raise ValueError(f"Invalid learning rate: {beta1}")
        if not (0.0 <= beta1 <= 1.0):
            raise ValueError(f"Invalid learning rate: {beta2}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1-beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1-beta2))
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                step_size = lr / bias_correction1
                demon = (exp_avg_sq / bias_correction2).sqrt() + eps

                p.addcdiv_(exp_avg, demon, value=-step_size)

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
        return loss
    






