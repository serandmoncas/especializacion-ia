"""Grad-CAM (Selvaraju et al., 2017): qué regiones de la imagen empujan la clase elegida."""

import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """Mapa [N, H, W] en [0, 1] para la clase `target`, al tamaño de la entrada.

    Pondera cada canal de las activaciones de `layer` por el promedio espacial de su gradiente,
    suma, aplica ReLU (solo evidencia a favor de la clase) y normaliza por imagen.
    """
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad()
        x = (
            x.detach().clone().requires_grad_(True)
        )  # garantiza el grafo aunque el modelo esté congelado
        model(x)[:, target].sum().backward()
    finally:
        handle.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12)
    return (cam / peak[:, None, None]).detach()
