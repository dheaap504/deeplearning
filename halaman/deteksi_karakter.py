"""Halaman: deteksi karakter plat dengan model Faster R-CNN (.pth) milik pengguna."""
import pandas as pd
import streamlit as st
from PIL import Image

from src.detector import detect, draw, label_to_char, load_detector
from src.ui import card_title, file_hash, hero, pct, setup, stats

setup()
hero("PyTorch · Faster R-CNN", "Deteksi Karakter (Model Anda)",
     "Jalankan model Faster R-CNN ResNet-50 FPN yang Anda latih sendiri (mis. di Google Colab) untuk "
     "mendeteksi tiap karakter pada plat nomor.")


@st.cache_resource(max_entries=2, show_spinner="Memuat model…")
def get_model(model_bytes: bytes, num_classes: int, _key: str):
    return load_detector(model_bytes, num_classes)


with st.container(key="row_det"):
    c1, c2 = st.columns([1, 1.5], gap="medium")
    with c1, st.container(key="card_dset"):
        card_title("1. Model & pengaturan")
        mf = st.file_uploader("File model (.pth / .pt)", type=["pth", "pt"],
                              help="State dict atau checkpoint berisi 'model_state_dict' / 'state_dict'.")
        nc = st.number_input("Jumlah kelas (termasuk background)", 2, 100, 37,
                             help="Bawaan 37 = background + 0–9 + A–Z.")
        thr = st.slider("Ambang keyakinan", 0.0, 1.0, 0.5, 0.05)
        st.caption("Pemetaan label bawaan: 1–10 → angka 0–9, 11–36 → huruf A–Z. Sesuaikan `label_to_char` di "
                   "`src/detector.py` bila dataset Anda berbeda.")
        st.caption("Belum punya model? Lihat `scripts/save_model.py` untuk menyimpan model dari notebook pelatihan.")
    with c2, st.container(key="card_dimg"):
        card_title("2. Gambar plat")
        up = st.file_uploader("Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="det_img")
        if not up:
            st.info("Unggah model dan gambar untuk memulai deteksi.", icon=":material/info:")
        elif not mf:
            st.image(up, width=360)
            st.warning("Unggah file model terlebih dahulu.", icon=":material/warning:")
        else:
            image = Image.open(up).convert("RGB")
            data = mf.getvalue()
            try:
                model = get_model(data, int(nc), file_hash(data))
            except Exception as e:
                st.error(f"Model gagal dimuat — periksa jumlah kelas dan format file. Detail: {e}")
                st.stop()
            with st.spinner("Mendeteksi…"):
                boxes, labels, scores = detect(model, image, thr)
            st.image(draw(image, boxes, labels, scores), width="stretch", caption="Hasil deteksi")
            if len(boxes) == 0:
                st.warning("Tidak ada karakter terdeteksi. Coba turunkan ambang keyakinan.")
            else:
                order = boxes[:, 0].argsort()
                text = "".join(label_to_char(int(labels[i])) for i in order)
                if len(boxes) > 15:
                    st.warning(f"{len(boxes)} deteksi — jauh lebih banyak dari jumlah karakter plat (±5–9). Model "
                               "kemungkinan belum terlatih, jumlah kelas tidak cocok, atau ambang terlalu rendah.",
                               icon=":material/warning:")
                    text = text[:30] + "…"
                stats([("Teks (kiri → kanan)", text, ""), ("Karakter terdeteksi", len(boxes), ""),
                       ("Rata-rata keyakinan", pct(float(scores.mean())), "")])
                st.dataframe(pd.DataFrame([{"Urutan": n + 1, "Karakter": label_to_char(int(labels[i])),
                                            "Keyakinan": float(scores[i]),
                                            "Kotak [x1, y1, x2, y2]": [int(v) for v in boxes[i]]}
                                           for n, i in enumerate(order)]),
                             hide_index=True, width="stretch",
                             column_config={"Keyakinan": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
