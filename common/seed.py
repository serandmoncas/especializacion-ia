"""Semillas reproducibles para random, NumPy y (si está instalado) PyTorch."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = False) -> np.random.Generator:
    """Fija semillas globales y devuelve un `np.random.Generator` sembrado.

    Preferir el generador devuelto sobre `np.random.*` global: hace explícito
    de dónde sale la aleatoriedad y evita acoplamiento entre celdas.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - por compatibilidad con librerías que usan el estado global

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

    return np.random.default_rng(seed)
