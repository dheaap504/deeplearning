"""Pipeline pengenalan plat nomor kendaraan Indonesia dengan EasyOCR.

Tahapan: (1) deteksi area plat berbasis kontur (opsional) → (2) beberapa variasi pra-pemrosesan →
(3) OCR dengan daftar karakter A–Z/0–9 → (4) buang teks kecil (mis. masa berlaku) → (5) cocokkan ke
format plat Indonesia + koreksi salah baca yang umum (O↔0, I↔1, S↔5, B↔8, ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Kode wilayah plat nomor (perkiraan wilayah registrasi)
REGION = {
    "A": "Banten", "B": "DKI Jakarta & sekitarnya", "D": "Bandung Raya", "E": "Cirebon & sekitarnya",
    "F": "Bogor, Sukabumi, Cianjur", "G": "Pekalongan & Tegal", "H": "Semarang & sekitarnya",
    "K": "Pati, Kudus & sekitarnya", "L": "Surabaya", "M": "Madura", "N": "Malang & sekitarnya",
    "P": "Besuki (Jember, Banyuwangi, dll.)", "R": "Banyumas & sekitarnya", "S": "Bojonegoro & sekitarnya",
    "T": "Purwakarta, Karawang, Subang", "W": "Gresik & Sidoarjo", "Z": "Garut, Tasikmalaya & sekitarnya",
    "AA": "Kedu (Magelang & sekitarnya)", "AB": "DI Yogyakarta", "AD": "Surakarta & sekitarnya",
    "AE": "Madiun & sekitarnya", "AG": "Kediri & sekitarnya", "BA": "Sumatera Barat",
    "BB": "Sumatera Utara (bagian barat)", "BD": "Bengkulu", "BE": "Lampung", "BG": "Sumatera Selatan",
    "BH": "Jambi", "BK": "Sumatera Utara", "BL": "Aceh", "BM": "Riau", "BN": "Kepulauan Bangka Belitung",
    "BP": "Kepulauan Riau", "DA": "Kalimantan Selatan", "DB": "Sulawesi Utara", "DC": "Sulawesi Barat",
    "DD": "Sulawesi Selatan", "DE": "Maluku", "DG": "Maluku Utara", "DH": "Nusa Tenggara Timur (Timor)",
    "DK": "Bali", "DL": "Sulawesi Utara (Sangihe & Talaud)", "DM": "Gorontalo", "DN": "Sulawesi Tengah",
    "DR": "Nusa Tenggara Barat (Lombok)", "DT": "Sulawesi Tenggara", "EA": "Nusa Tenggara Barat (Sumbawa)",
    "EB": "Nusa Tenggara Timur (Flores)", "ED": "Nusa Tenggara Timur (Sumba)", "KB": "Kalimantan Barat",
    "KH": "Kalimantan Tengah", "KT": "Kalimantan Timur", "KU": "Kalimantan Utara", "PA": "Papua",
    "PB": "Papua Barat",
}

# Koreksi salah baca: angka yang terbaca sebagai huruf (di bagian huruf) dan sebaliknya
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1", "Z": "2", "A": "4",
            "S": "5", "G": "6", "T": "7", "B": "8"}


@dataclass
class PlateParse:
    raw: str
    formatted: str
    valid: bool
    region_code: str = ""
    region: str = ""
    corrections: int = 0


@dataclass
class PlateResult:
    success: bool = False
    raw_text: str = ""
    parse: PlateParse | None = None
    confidence: float = 0.0
    method: str = ""
    plate_img: Image.Image | None = None
    viz_img: Image.Image | None = None
    plate_found: bool | None = None
    tokens: list = field(default_factory=list)
    attempts: dict = field(default_factory=dict)
    error: str = ""


# ============================================================ format plat
def _convert(part: str, table: dict, want_alpha: bool) -> tuple[str, int] | None:
    out, n = [], 0
    for ch in part:
        ok = ch.isalpha() if want_alpha else ch.isdigit()
        if ok:
            out.append(ch)
        elif ch in table:
            out.append(table[ch])
            n += 1
        else:
            return None
    return "".join(out), n


def parse_plate(text: str) -> PlateParse:
    """Cocokkan teks ke pola plat Indonesia: [1–2 huruf][1–4 angka][0–3 huruf]."""
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    best = None
    for p in (1, 2):
        for d in range(1, 5):
            s = len(clean) - p - d
            if not 0 <= s <= 3:
                continue
            a = _convert(clean[:p], TO_LETTER, True)
            b = _convert(clean[p:p + d], TO_DIGIT, False)
            c = _convert(clean[p + d:], TO_LETTER, True)
            if not (a and b and c is not None):
                continue
            fixes = a[1] + b[1] + c[1]
            known = a[0] in REGION
            # penalti: tiap koreksi karakter, kode wilayah tak dikenal, angka < 4 digit (4 digit paling umum)
            score = fixes + (0 if known else 1.2) + (4 - d) * 0.4
            if best is None or score < best[0]:
                best = (score, a[0], b[0], c[0], fixes)
    if best is None:
        # teks masa berlaku (mis. "0527") kadang ikut terbaca di belakang → coba buang
        m = re.match(r"^(.*?[A-Z])\d{3,4}$", clean)
        if m and len(m.group(1)) >= 2:
            p2 = parse_plate(m.group(1))
            if p2.valid:
                p2.raw = clean
                return p2
        return PlateParse(raw=clean, formatted=clean, valid=False)
    _, a, b, c, fixes = best
    formatted = " ".join(x for x in (a, b, c) if x)
    return PlateParse(raw=clean, formatted=formatted, valid=True, region_code=a,
                      region=REGION.get(a, "Kode wilayah tidak dikenal"), corrections=fixes)


# ============================================================ pra-pemrosesan
def _gray(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
    h, w = g.shape
    if h < 100:  # perbesar plat kecil agar karakter terbaca
        g = cv2.resize(g, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    return g


def preprocess(image: Image.Image, method: str) -> np.ndarray:
    """method: original | auto | white_plate | black_plate"""
    gray = _gray(image)
    if method == "original":
        return gray
    if method == "auto":
        method = "black_plate" if gray.mean() < 127 else "white_plate"
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if method == "black_plate":  # plat hitam teks putih → balik jadi teks hitam
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


def pad(arr: np.ndarray) -> np.ndarray:
    """Tambah tepi kosong agar huruf yang mepet ke tepi gambar tetap terdeteksi."""
    edge = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    p = int(0.15 * arr.shape[0]) + 8
    return cv2.copyMakeBorder(arr, p, p, p, p, cv2.BORDER_CONSTANT, value=int(np.median(edge)))


# urutan percobaan: berhenti lebih awal bila hasil sudah meyakinkan (hemat waktu di CPU)
METHODS = ["auto", "original", "white_plate", "black_plate"]
METHOD_LABEL = {"original": "Grayscale asli", "auto": "Otomatis", "white_plate": "Plat putih",
                "black_plate": "Plat hitam"}


# ============================================================ deteksi area plat
def detect_plate_region(image: Image.Image):
    """Cari kontur persegi dengan rasio lebar/tinggi 2–5. Return (crop, box) atau (image, None)."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0 or w * h < 0.002 * W * H:
            continue
        if 2.0 <= w / h <= 5.5:
            pad = int(0.04 * w)
            box = (max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad))
            return image.crop(box), box
    return image, None


def draw_box(image: Image.Image, box, label: str) -> Image.Image:
    img = image.copy()
    d = ImageDraw.Draw(img)
    lw = max(3, image.width // 200)
    d.rectangle(box, outline=(67, 24, 255), width=lw)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, image.width // 40))
    except OSError:
        font = ImageFont.load_default()
    tb = d.textbbox((box[0], box[1]), label, font=font)
    th = tb[3] - tb[1] + 10
    y = box[1] - th if box[1] - th > 0 else box[3]
    d.rectangle((box[0], y, box[0] + tb[2] - tb[0] + 14, y + th), fill=(67, 24, 255))
    d.text((box[0] + 7, y + 3), label, fill="white", font=font)
    return img


# ============================================================ OCR
def _read(reader, arr: np.ndarray, height_ratio: float):
    res = reader.readtext(pad(arr), allowlist=PLATE_ALLOWLIST, paragraph=False)
    items = []
    for bbox, text, conf in res:
        t = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not t:
            continue
        h = ((bbox[3][1] - bbox[0][1]) + (bbox[2][1] - bbox[1][1])) / 2
        items.append(dict(text=t, conf=float(conf), height=float(h), x=float((bbox[0][0] + bbox[2][0]) / 2),
                          bbox=[[float(p[0]), float(p[1])] for p in bbox]))
    if not items:
        return "", 0.0, []
    tallest = max(i["height"] for i in items)
    keep = sorted([i for i in items if i["height"] >= tallest * height_ratio], key=lambda i: i["x"])
    text = "".join(i["text"] for i in keep)
    conf = float(np.mean([i["conf"] for i in keep])) if keep else 0.0
    return text, conf, keep


def recognize(reader, image: Image.Image, detect_plate: bool = False, height_ratio: float = 0.6,
              early_stop: float = 0.85) -> PlateResult:
    r = PlateResult()
    try:
        plate, box = (detect_plate_region(image) if detect_plate else (image, None))
        r.plate_img = plate
        r.plate_found = (box is not None) if detect_plate else None
        best = None
        for m in METHODS:
            try:
                text, conf, tokens = _read(reader, preprocess(plate, m), height_ratio)
            except Exception as e:  # satu metode gagal tidak menghentikan yang lain
                r.attempts[m] = dict(error=str(e))
                continue
            p = parse_plate(text)
            r.attempts[m] = dict(text=text, confidence=round(conf, 4), valid_format=p.valid)
            # pilih: format valid > tidak valid, lalu keyakinan tertinggi
            key = (p.valid and len(p.raw) >= 3, conf)
            if text and (best is None or key > best[0]):
                best = (key, m, text, conf, tokens, p)
            if p.valid and conf >= early_stop:
                break
        if best:
            _, r.method, r.raw_text, r.confidence, r.tokens, r.parse = best
            r.success = True
        r.viz_img = draw_box(image, box, r.parse.formatted if r.parse else "Plat") if box else image
    except Exception as e:
        r.error = str(e)
    return r
