"""Arsitektur CNN untuk klasifikasi CIFAR-10 (sama persis dengan versi awal repo)."""
from __future__ import annotations

import torch.nn as nn

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASSES_ID = {
    "airplane": "Pesawat", "automobile": "Mobil", "bird": "Burung", "cat": "Kucing", "deer": "Rusa",
    "dog": "Anjing", "frog": "Katak", "horse": "Kuda", "ship": "Kapal", "truck": "Truk",
}


class CNNClassifier(nn.Module):
    """3 blok konvolusi (Conv → ReLU → MaxPool) + 2 lapisan fully connected."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu4(self.fc1(x)))
        return self.fc2(x)
