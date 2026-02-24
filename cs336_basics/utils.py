import os
from tabnanny import verbose
import torch
import typing

from rich import print

def print_color(content: str, color:str = "green"):
    print(f"[{color}]{content}[/{color}]")

def get_device(verbose: bool = True):
    if torch.cuda.is_available():
        if verbose:
            print_color("using CUDA device", "blue")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")

def get_dataset_memmap(path, dtype=np.uint16):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration,
    out: str | os.PathLike | typing.IO[bytes] | typing.BinaryIO,
    ):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(state, out)

    if verbose:
        print_color(f"Checkpoint saved to {out}", "blue")
def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.nn.Module
    )->int:
    state = torch.load(src, map_location=get_device())
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if verbose:
        print_color(f"Checkpoint loaded from {src}", "blue")

    return state["iteration"]