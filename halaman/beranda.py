"""Halaman beranda: ringkasan semua alat."""
import streamlit as st

from src.cnn_tools import load_metrics
from src.ui import MODELS, hero, pill, setup

setup()
hero("Proyek Deep Learning", "Streamlit Deep Learning",
     "Kumpulan aplikasi computer vision berbasis PyTorch & EasyOCR: membaca plat nomor kendaraan Indonesia, "
     "mengklasifikasikan gambar, dan menjalankan model deteksi karakter buatan sendiri.")

met = load_metrics()
acc = f"{met['test_accuracy']:.1f}%".replace(".", ",") if met else "–"
tools = [
    ("🚗", "Pengenalan Plat Nomor", "halaman/plat_nomor.py", pill("Siap dipakai", "ok"),
     "Unggah foto atau gunakan kamera. EasyOCR membaca teks, lalu sistem mencocokkan pola plat Indonesia "
     "(mis. B 1234 XYZ), mengoreksi salah baca umum, dan menebak wilayah registrasi.",
     ["Banyak gambar sekaligus", "Deteksi area plat otomatis", "Unduh hasil CSV"]),
    ("🖼️", "Klasifikasi Gambar (CNN)", "halaman/klasifikasi_gambar.py",
     pill(f"Akurasi uji {acc}", "ok") if (MODELS / "cnn_cifar10.pth").exists() else pill("Perlu dilatih", "warn"),
     "CNN 3 blok konvolusi dilatih dari nol pada 50.000 gambar CIFAR-10 untuk mengenali 10 jenis objek: "
     "pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, dan truk.",
     ["Probabilitas tiap kelas", "Evaluasi & confusion matrix", "Latih ulang / unggah bobot"]),
    ("🎯", "Deteksi Karakter (Model Anda)", "halaman/deteksi_karakter.py", pill("Butuh file .pth", "info"),
     "Unggah model Faster R-CNN ResNet-50 FPN yang Anda latih sendiri untuk mendeteksi setiap karakter "
     "pada plat beserta kotak dan keyakinannya.",
     ["Mendukung state dict & checkpoint", "Urutan karakter kiri → kanan", "Jumlah kelas dapat diatur"]),
]
with st.container(key="row_tools"):
    cols = st.columns(3, gap="medium")
    for col, (ic, title, page, badge, desc, feats) in zip(cols, tools):
        with col, st.container(key=f"card_{page.split('/')[-1][:-3]}"):
            st.markdown(f'<div class="tool"><div class="ic">{ic}</div><p class="card-t">{title}</p>'
                        f'<div style="margin:.35rem 0 .6rem">{badge}</div><p class="card-s">{desc}</p>'
                        + "".join(f"<div style='font-size:.85rem;color:#374151;margin:.2rem 0'>✓ {f}</div>" for f in feats)
                        + "</div>", unsafe_allow_html=True)
            st.page_link(page, label="Buka", icon=":material/arrow_forward:")

st.write("")
with st.expander("Tentang proyek & teknologi"):
    st.markdown("""
- **Framework:** Streamlit, PyTorch, torchvision, EasyOCR, OpenCV, Plotly
- **Inferensi:** CPU (tanpa GPU) — cocok untuk Streamlit Community Cloud gratis
- **Catatan:** pertama kali halaman plat nomor dibuka, model EasyOCR (±100 MB) diunduh otomatis, jadi
  butuh ±1 menit. Setelah itu proses jauh lebih cepat.
- Kode eksperimen ANPR versi awal (pipeline Faster R-CNN tanpa bobot terlatih) disimpan di folder `arsip/`.
""")
