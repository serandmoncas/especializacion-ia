"""Bucle de entrenamiento en PyTorch con early stopping. Requiere el extra `dl`."""

import copy
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device(prefer_accelerator: bool = True) -> torch.device:
    """CUDA o MPS si hay y se pide; si no, CPU. Para resultados exactamente reproducibles, CPU."""
    if prefer_accelerator:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, loss_fn=None
) -> tuple[float, float]:
    """(pérdida media, accuracy) en modo evaluación (sin dropout, BN con estadísticas fijas)."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    seconds: float = 0.0


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 20,
    patience: int = 5,
    device: torch.device | None = None,
    scheduler=None,
    loss_fn=None,
    train_transform=None,
    verbose: bool = False,
) -> History:
    """Entrena y se queda con los pesos de la época de menor pérdida de validación.

    Early stopping: si la validación no mejora en `patience` épocas, se detiene. Es una forma de
    regularización (limita cuánto se aleja el modelo de la inicialización) que no cuesta nada.

    `train_transform(xb) -> xb` se aplica solo a los lotes de entrenamiento (aumentación de datos),
    ya en el dispositivo: la validación ve siempre las imágenes originales.
    """
    device = device or get_device(False)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.to(device)
    hist, best_state, best_val, stale = History(), None, float("inf"), 0
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if train_transform is not None:
                xb = train_transform(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
        if scheduler is not None:
            scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        hist.train_loss.append(running / n)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
        if verbose:
            print(
                f"época {epoch + 1:>3}: train {running / n:.4f}"
                f"  val {val_loss:.4f}  acc {val_acc:.4f}"
            )
        if val_loss < best_val - 1e-6:
            best_val, stale, hist.best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist.seconds = time.perf_counter() - start
    return hist
