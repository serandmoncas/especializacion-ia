"""Detector de mazorcas (una sola clase): Faster R-CNN MobileNetV3-FPN de torchvision.

Separar «encontrar la mazorca» de «diagnosticar la etapa» permite ver dónde falla el sistema: el
detector solo responde «¿hay una mazorca aquí?»; la etapa la decide el clasificador del recorte.
"""

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


def build_detector(pretrained: bool, min_size: int = 512, max_size: int = 640) -> nn.Module:
    """Pesos de COCO (o al azar) y una cabeza nueva de 2 clases: fondo y mazorca.

    `min_size`/`max_size`: el modelo reescala cada foto por dentro; 512–640 px entrenan en ~2–3 min
    por época en un M4 (MPS).
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


class DetectionDataset(Dataset):
    """Fotos con sus cajas; con `flip_prob` > 0 voltea horizontalmente foto y cajas a la vez."""

    def __init__(
        self, paths: list, boxes: list[np.ndarray], flip_prob: float = 0.0, seed: int = 42
    ):
        self.paths, self.boxes, self.flip_prob = list(paths), list(boxes), flip_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            image = to_tensor(im.convert("RGB"))
        boxes = torch.as_tensor(np.asarray(self.boxes[i], dtype=np.float32).reshape(-1, 4))
        if self.flip_prob and self.rng.random() < self.flip_prob:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
            )
        return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch, strict=True))


def train_detector(
    model: nn.Module,
    ds: Dataset,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 0.005,
    batch_size: int = 4,
    seed: int = 42,
) -> list[float]:
    """SGD con momentum y tasa coseno; devuelve la pérdida total media de cada época."""
    torch.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    model.to(device).train()
    history = []
    for _ in range(epochs):
        total = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        history.append(total / len(loader))
    return history


@torch.no_grad()
def detect(
    model: nn.Module, paths: list, device: torch.device, score_threshold: float = 0.05
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(cajas, puntajes) por foto, ya con NMS (lo aplica el propio Faster R-CNN)."""
    model.to(device).eval()
    results = []
    for path in paths:
        with Image.open(path) as im:
            x = to_tensor(im.convert("RGB")).to(device)
        pred = model([x])[0]
        keep = pred["scores"] >= score_threshold
        results.append((pred["boxes"][keep].cpu().numpy(), pred["scores"][keep].cpu().numpy()))
    return results
