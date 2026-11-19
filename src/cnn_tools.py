"""Muat, prediksi, dan latih ulang CNN CIFAR-10."""
from __future__ import annotations

import io
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms

from .cnn_model import CLASSES, CNNClassifier
from .ui import MODELS, ROOT

WEIGHTS = MODELS / "cnn_cifar10.pth"
METRICS = MODELS / "cnn_cifar10_metrics.json"
NORM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
INFER_TF = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), NORM])


def load_pretrained() -> tuple[CNNClassifier, bool]:
    m = CNNClassifier(len(CLASSES))
    ok = WEIGHTS.exists()
    if ok:
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()
    return m, ok


def load_metrics() -> dict | None:
    return json.loads(METRICS.read_text()) if METRICS.exists() else None


def load_from_bytes(data: bytes) -> CNNClassifier:
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    m = CNNClassifier(len(CLASSES))
    m.load_state_dict(state)
    m.eval()
    return m


def to_bytes(model: CNNClassifier) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def predict(model: CNNClassifier, image: Image.Image) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(INFER_TF(image.convert("RGB")).unsqueeze(0))
        return torch.softmax(logits, dim=1)[0].numpy()


def train(model: CNNClassifier, epochs: int, batch_size: int, lr: float, on_progress=None) -> list[dict]:
    """Latih pada CIFAR-10 (diunduh otomatis ±170 MB ke folder data/ saat pertama kali)."""
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                             transforms.ToTensor(), NORM])
    ds = datasets.CIFAR10(root=str(ROOT / "data"), train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), NORM]))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    tl = torch.utils.data.DataLoader(test, batch_size=500, num_workers=0)
    crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)
    hist, steps = [], epochs * len(dl)
    for ep in range(epochs):
        model.train()
        loss_sum = correct = n = 0
        for i, (x, y) in enumerate(dl):
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
            if on_progress and i % 20 == 0:
                on_progress((ep * len(dl) + i) / steps, f"Epoch {ep + 1}/{epochs} · batch {i}/{len(dl)} · "
                                                         f"loss {loss_sum / n:.3f} · akurasi latih {correct / n:.1%}")
        model.eval()
        tc = tn = 0
        with torch.no_grad():
            for x, y in tl:
                tc += (model(x).argmax(1) == y).sum().item()
                tn += len(y)
        hist.append(dict(epoch=ep + 1, loss=loss_sum / n, train_acc=correct / n * 100, test_acc=tc / tn * 100))
    model.eval()
    return hist
