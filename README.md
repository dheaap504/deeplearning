# Streamlit Deep Learning

Aplikasi computer vision berbasis **PyTorch** dan **EasyOCR** yang dibangun dengan **Streamlit**.
Satu aplikasi berisi tiga alat:

| Halaman | Fungsi | Model |
|---|---|---|
| **Pengenalan Plat Nomor** | Membaca plat nomor kendaraan Indonesia dari foto (unggah, kamera, atau gambar contoh), mencocokkan pola `B 1234 XYZ`, mengoreksi salah baca umum (O↔0, I↔1, …), dan menebak wilayah registrasi | EasyOCR (pralatih) + OpenCV |
| **Klasifikasi Gambar (CNN)** | Mengenali 10 jenis objek CIFAR-10 beserta probabilitasnya; dilengkapi evaluasi (akurasi per kelas, confusion matrix, kurva latih), latih ulang, serta unduh/unggah bobot | CNN kustom 3 blok konvolusi, dilatih dari nol (`models/cnn_cifar10.pth`) |
| **Deteksi Karakter (Model Anda)** | Menjalankan model Faster R-CNN ResNet-50 FPN buatan sendiri (`.pth`) untuk mendeteksi tiap karakter plat | Faster R-CNN (diunggah pengguna) |

Semua inferensi berjalan di **CPU**, sehingga bisa di-deploy gratis di Streamlit Community Cloud.

## Struktur proyek

```
├── app.py                      # Titik masuk: navigasi multi-halaman
├── halaman/
│   ├── beranda.py
│   ├── plat_nomor.py           # Pengenalan plat nomor (EasyOCR)
│   ├── klasifikasi_gambar.py   # Klasifikasi CNN CIFAR-10
│   └── deteksi_karakter.py     # Faster R-CNN milik pengguna
├── src/
│   ├── plate_ocr.py            # Pipeline OCR plat + format plat Indonesia + kode wilayah
│   ├── cnn_model.py            # Arsitektur CNN
│   ├── cnn_tools.py            # Muat, prediksi, latih ulang CNN
│   ├── detector.py             # Muat & jalankan Faster R-CNN
│   └── ui.py                   # Gaya visual bersama
├── models/
│   ├── cnn_cifar10.pth         # Bobot CNN pralatih (~2,5 MB)
│   └── cnn_cifar10_metrics.json
├── assets/contoh/              # Gambar contoh (plat sintetis & sampel uji CIFAR-10)
├── scripts/                    # save_model.py, inference_example.py (dipakai di notebook/Colab)
├── arsip/                      # Eksperimen ANPR versi awal (tidak ikut di-deploy, lihat arsip/README.md)
├── .streamlit/config.toml      # Tema & batas ukuran unggahan
├── .devcontainer/              # GitHub Codespaces
└── requirements.txt
```

## Menjalankan secara lokal

Butuh Python 3.10–3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Saat halaman plat nomor pertama kali dibuka, EasyOCR mengunduh modelnya (±100 MB, sekali saja).

## Deploy ke Streamlit Community Cloud (gratis)

1. Pastikan semua file di atas sudah ada di repo GitHub (termasuk `models/` dan `.streamlit/config.toml`).
2. Buka **share.streamlit.io** → **Create app** → **Deploy a public app from GitHub**.
3. Isi: repository ini, branch `main`, **Main file path: `app.py`**.
4. **Advanced settings → Python version: 3.12** → **Deploy**.
5. Instalasi pertama ±5–10 menit (PyTorch + EasyOCR). Setelah itu aplikasi siap dipakai.

`requirements.txt` memakai PyTorch versi **CPU** (`--extra-index-url https://download.pytorch.org/whl/cpu`)
yang jauh lebih kecil dari versi GPU, agar muat dalam batas memori Community Cloud.

### Jika deploy gagal
| Gejala | Solusi |
|---|---|
| Instalasi kehabisan memori / waktu | Reboot app dari **Manage app → ⋮ → Reboot** |
| `ModuleNotFoundError: src` | Pastikan folder `src/` (berisi `__init__.py`) ada di repo |
| Halaman plat lama saat pertama dibuka | Normal — model EasyOCR sedang diunduh |
| Model `.pth` > 200 MB ditolak saat diunggah | Batas diatur di `.streamlit/config.toml` (`maxUploadSize = 400`) |

## Catatan model

- **CNN CIFAR-10:** dilatih 20 epoch pada 50.000 gambar; akurasi uji tercantum di halaman *Evaluasi model*.
  Model kecil ini dibuat untuk pembelajaran — paling cocok untuk foto satu objek di tengah gambar.
- **OCR plat:** hasil terbaik pada foto plat yang tajam dan lurus. Mode *Foto kendaraan utuh* mencari area plat
  dengan deteksi kontur sederhana; bila gagal, seluruh gambar dibaca.
- Kode wilayah plat adalah perkiraan berdasarkan huruf depan plat.
