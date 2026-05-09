"""
Utility functions for 3DisoDeepPF
"""

import torch
import random
import numpy as np
from typing import Optional


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, name: str = "Model"):
    """Print model information."""
    num_params = count_parameters(model)
    print(f"\n{name} Information:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable: {num_params:,}")
    print(f"  Device: {next(model.parameters()).device}")
