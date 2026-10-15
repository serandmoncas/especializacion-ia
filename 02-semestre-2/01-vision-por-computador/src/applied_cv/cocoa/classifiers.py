"""Clasificadores de etapa sobre recortes: logística sobre rasgos y ResNet-18."""

from pathlib import Path

import numpy as np
import torch
from applied_dl.training import History, fit
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def train_augmentation() -> transforms.Compose:
    """Aumentación del clasificador. Restricción de dominio: no cambiar el tono (test lo verifica).

    TODO(usuario): esta es la versión mínima (solo volteo). Decide qué transformaciones conservan el
    síntoma: ¿rotaciones (la mazorca puede estar inclinada)? ¿brillo y contraste (sol y sombra en
    campo)? ¿recortes aleatorios (que podrían cortar la joroba de una m1)?
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CropDataset(Dataset):
    """Recortes desde archivo o en memoria (los del detector), con etiqueta opcional."""

    def __init__(self, items: list, labels: list[int] | None, transform):
        self.items, self.labels, self.transform = list(items), labels, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        if isinstance(item, str | Path):
            with Image.open(item) as im:
                image = im.convert("RGB")
        else:
            image = item.convert("RGB")
        x = self.transform(image)
        return x, (self.labels[i] if self.labels is not None else -1)


def build_resnet(n_classes: int, pretrained: bool) -> nn.Module:
    """ResNet-18 con pesos de ImageNet (o al azar) y una capa final nueva de `n_classes`."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def fine_tune(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 32,
) -> History:
    """Ajusta toda la red con AdamW y early stopping sobre la pérdida de validación."""
    gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return fit(model, train_loader, val_loader, optimizer, epochs=epochs, patience=3, device=device)


@torch.no_grad()
def predict_proba(
    model: nn.Module, ds: Dataset, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    model.to(device).eval()
    out = [
        torch.softmax(model(x.to(device)), dim=1).cpu()
        for x, _ in DataLoader(ds, batch_size=batch_size)
    ]
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_features(
    model: nn.Module, ds: Dataset, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Vector de 512 de la penúltima capa (sonda lineal y H3). Restaura `fc` al terminar."""
    head, model.fc = model.fc, nn.Identity()
    try:
        model.to(device).eval()
        feats = [model(x.to(device)).cpu() for x, _ in DataLoader(ds, batch_size=batch_size)]
    finally:
        model.fc = head
    return torch.cat(feats).numpy()


class LogisticModel:
    """Escalado + logística multinomial con columnas para todas las clases."""

    def __init__(self, pipeline, n_classes: int, C: float):
        self.pipeline, self.n_classes, self.C = pipeline, n_classes, C

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), self.n_classes))
        proba[:, self.pipeline.classes_] = self.pipeline.predict_proba(X)
        return proba


def fit_logistic(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_classes: int = 4,
    Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> LogisticModel:
    """Elige C por F1-macro en validación; clases balanceadas (h0 es la más frecuente)."""
    best = None
    for C in Cs:
        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        ).fit(X_tr, y_tr)
        score = f1_score(y_va, pipe.predict(X_va), average="macro")
        if best is None or score > best[0]:
            best = (score, C, pipe)
    return LogisticModel(best[2], n_classes, best[1])
