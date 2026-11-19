# Arsip eksperimen ANPR (tidak ikut di-deploy)

File di folder ini adalah versi awal pipeline ANPR (Faster R-CNN + CNN karakter):

- `anpr_app.py` – versi dasar
- `anpr_enhanced.py` – versi dengan tampilan lebih lengkap

**Kenapa diarsipkan:** detektor plat Faster R-CNN di kedua file dibuat dengan `weights=None` dan tidak pernah
dilatih, sehingga kotak deteksinya acak. Pengklasifikasi karakter hanya dilatih singkat pada MNIST (angka 0–9)
setiap kali aplikasi dibuka, jadi huruf A–Z juga acak. Fungsi pengenalan plat di aplikasi utama sekarang
memakai EasyOCR (`halaman/plat_nomor.py`), yang sudah terlatih dan memberi hasil nyata.

Untuk menjalankannya secara lokal (membutuhkan internet untuk mengunduh MNIST):
```bash
streamlit run arsip/anpr_enhanced.py
```
