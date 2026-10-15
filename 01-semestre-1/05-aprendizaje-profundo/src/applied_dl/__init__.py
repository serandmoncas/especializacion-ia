"""Aprendizaje profundo: autograd y redes desde cero (NumPy) y entrenamiento en PyTorch.

`autograd`, `nn` y `conv` no dependen de PyTorch (se verifican contra él en los tests).
`training` sí lo requiere (extra `dl`).
"""

from applied_dl.autograd import Tensor, cross_entropy

__all__ = ["Tensor", "cross_entropy"]
