"""U-Net pequeña (Ronneberger et al., 2015): codificador, decodificador y conexiones de salto."""

import torch
from torch import nn


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Dos niveles de reducción. Las conexiones de salto devuelven al decodificador el detalle
    espacial que el codificador perdió al reducir: así los bordes de la máscara salen nítidos."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2, base: int = 8):
        super().__init__()
        self.down1 = _double_conv(in_channels, base)
        self.down2 = _double_conv(base, 2 * base)
        self.bottom = _double_conv(2 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = _double_conv(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = _double_conv(2 * base, base)
        self.head = nn.Conv2d(base, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError(f"H y W deben ser múltiplos de 4; llegó {tuple(x.shape[-2:])}")
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        b = self.bottom(self.pool(d2))
        u2 = self.dec2(torch.cat([self.up2(b), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)
