
import torch
import numpy as np

def data_loading(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device = None,
)-> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    x_tensor = torch.as_tensor(x)

    start = torch.randint(0, n - context_length, (batch_size,), dtype = torch.long) # (B,)
    offsets = torch.arange(context_length, dtype=torch.long).unsqueeze(0) # (1,m)
    idx = start.unsqueeze(1) + offsets # (B, m)

    input =x_tensor[idx] 
    target = x_tensor[idx+1]

    input = input.to(device)
    target=target.to(device)

    return input, target