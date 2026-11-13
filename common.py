"""Shared model definition and CIFAR-10 loading."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CKPT = MODEL_DIR / "cnn_cifar10.pt"
ONNX_FP32 = MODEL_DIR / "cnn_fp32.onnx"
ONNX_INT8 = MODEL_DIR / "cnn_int8.onnx"  # 8-bit weights
ONNX_INT8_RR = MODEL_DIR / "cnn_int8_rr.onnx"  # 7-bit weights (--reduce-range)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def load_cifar10(train: bool):
    """Return (x, y): x is normalized float32 NCHW, y is int64 labels."""
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(DATA_DIR), train=train, download=True)
    x = ds.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = (x - MEAN) / STD
    y = np.asarray(ds.targets, dtype=np.int64)
    return np.ascontiguousarray(x), y


class Block(nn.Module):
    """Two 3x3 conv-BN-ReLU layers with a residual connection."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """ResNet-8 style CNN for 32x32 inputs (~0.3M parameters)."""

    def __init__(self, num_classes: int = 10, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = Block(widths[0], widths[0], 1)
        self.stage2 = Block(widths[0], widths[1], 2)
        self.stage3 = Block(widths[1], widths[2], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.fc(torch.flatten(self.pool(x), 1))
