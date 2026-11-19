"""Muat & jalankan model Faster R-CNN (ResNet-50 FPN) untuk deteksi karakter plat."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(model_bytes: bytes, num_classes: int = 37):
    ckpt = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        state = ckpt
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model = build(num_classes)
        model.load_state_dict(state)
    model.eval()
    return model


def label_to_char(label: int) -> str:
    """0 = background, 1–10 = angka 0–9, 11–36 = huruf A–Z (sesuaikan dengan dataset Anda)."""
    if 1 <= label <= 10:
        return str(label - 1)
    if 11 <= label <= 36:
        return chr(ord("A") + label - 11)
    return "?" if label == 0 else str(label)


def detect(model, image: Image.Image, threshold: float = 0.5):
    with torch.no_grad():
        pred = model([transforms.ToTensor()(image)])[0]
    boxes, labels, scores = (pred[k].cpu().numpy() for k in ("boxes", "labels", "scores"))
    keep = scores >= threshold
    return boxes[keep], labels[keep], scores[keep]


def draw(image: Image.Image, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> Image.Image:
    img = image.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, image.width // 45))
    except OSError:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
        d.rectangle([(x1, y1), (x2, y2)], outline=(79, 70, 229), width=3)
        t = f"{label_to_char(int(lab))} {sc:.2f}"
        tb = d.textbbox((x1, y1), t, font=font)
        h = tb[3] - tb[1] + 6
        y = y1 - h if y1 - h > 0 else y2
        d.rectangle((x1, y, x1 + tb[2] - tb[0] + 8, y + h), fill=(79, 70, 229))
        d.text((x1 + 4, y + 2), t, fill="white", font=font)
    return img
