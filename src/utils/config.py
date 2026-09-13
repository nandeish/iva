"""
Small helper for loading the project YAML config and setting global seeds.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load the master YAML config into a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_global_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preferred: str = "cuda") -> "torch.device":  # noqa: F821
    import torch

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
