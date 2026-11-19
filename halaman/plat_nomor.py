"""Halaman: pengenalan plat nomor kendaraan Indonesia (EasyOCR)."""
import html
import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.plate_ocr import METHOD_LABEL, recognize
from src.ui import SAMPLES, card_title, file_hash, hero, pct, pill, setup, stats

setup()
hero("EasyOCR · OpenCV", "Pengenalan Plat Nomor",
     "Baca nomor plat kendaraan Indonesia dari foto — lengkap dengan perkiraan wilayah registrasi.")


@st.cache_resource(show_spinner="Menyiapkan model OCR… (pertama kali ±1 menit untuk mengunduh model)")
def get_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


@st.cache_data(show_spinner=False, max_entries=64)
def run_ocr(img_bytes: bytes, detect_plate: bool, height_ratio: float):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return recognize(get_reader(), img, detect_plate=detect_plate, height_ratio=height_ratio)


SAMPLE_FILES = {
    "Plat hitam (B 1234 XYZ)": ("plat_hitam_B1234XYZ.png", "crop"),
    "Plat putih (D 4521 AB)": ("plat_putih_D4521AB.png", "crop"),
    "Foto mobil (AB 7788 KC)": ("mobil_AB7788KC.jpg", "full"),
}

# ------------------------------------------------------------------ input
with st.container(key="row_in"):
    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, st.container(key="card_input"):
        card_title("1. Masukkan gambar", "unggah satu atau banyak foto, gunakan kamera, atau coba gambar contoh")
        src = st.segmented_control("Sumber", ["Unggah file", "Kamera", "Gambar contoh"], default="Unggah file",
                                   key="ocr_src", label_visibility="collapsed") or "Unggah file"
        images: list[tuple[str, bytes]] = []
        auto_mode = None
        if src == "Unggah file":
            files = st.file_uploader("Foto kendaraan atau plat (JPG/PNG)", type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True)
            images = [(f.name, f.getvalue()) for f in files or []]
        elif src == "Kamera":
            st.caption("Jika kamera tidak muncul, izinkan akses kamera di browser (ikon di dekat alamat situs).")
            cam = st.camera_input("Ambil foto plat", label_visibility="collapsed")
            if cam:
                images = [("foto-kamera.jpg", cam.getvalue())]
        else:
            pick = st.pills("Pilih contoh", list(SAMPLE_FILES), default=list(SAMPLE_FILES)[0], key="ocr_sample")
            if pick:
                fn, kind = SAMPLE_FILES[pick]
                images = [(fn, (SAMPLES / fn).read_bytes())]
                auto_mode = kind
                st.image(str(SAMPLES / fn), width=360)
            st.caption("Gambar contoh dibuat secara sintetis untuk demonstrasi.")
    with c2, st.container(key="card_set"):
        card_title("2. Pengaturan")
        kind = st.radio("Jenis gambar", ["Plat sudah dipotong (crop)", "Foto kendaraan utuh"],
                        index=1 if auto_mode == "full" else 0, key=f"ocr_kind_{auto_mode}",
                        captions=["Gambar hanya berisi plat", "Sistem mencari area plat lebih dulu"])
        thr = st.slider("Ambang keyakinan", 0.0, 1.0, 0.5, 0.05,
                        help="Hasil di bawah ambang ini ditandai 'perlu dicek'.")
        with st.expander("Lanjutan"):
            hr = st.slider("Filter tinggi huruf", 0.3, 0.9, 0.6, 0.05,
                           help="Teks yang lebih pendek dari rasio ini terhadap huruf tertinggi diabaikan "
                                "(membuang tulisan kecil seperti masa berlaku).")
        run = st.button(f"Baca {len(images)} plat" if len(images) > 1 else "Baca plat nomor", type="primary",
                        icon=":material/search:", width="stretch", disabled=not images)

detect = kind.startswith("Foto")

# ------------------------------------------------------------------ proses
if run and images:
    get_reader()
    results = []
    bar = st.progress(0.0, text="Memproses…")
    for i, (name, data) in enumerate(images):
        bar.progress(i / len(images), text=f"Memproses {name} ({i + 1}/{len(images)})…")
        results.append((name, data, run_ocr(data, detect, hr)))
    bar.empty()
    st.session_state.ocr_results = dict(items=results, detect=detect)

res = st.session_state.get("ocr_results")
if not res:
    st.info("Masukkan gambar lalu klik **Baca plat nomor**. Belum punya foto? Pilih **Gambar contoh**.", icon=":material/info:")
    st.stop()

items = res["items"]
ok = [r for _, _, r in items if r.success]
valid = [r for r in ok if r.parse and r.parse.valid]
sure = [r for r in valid if r.confidence >= thr]

st.write("")
with st.container(key="card_sum"):
    card_title("3. Hasil", f"{len(items)} gambar diproses")
    stats([
        ("Terbaca", f"{len(ok)}/{len(items)}", "teks ditemukan"),
        ("Format plat valid", f"{len(valid)}/{len(items)}", "pola huruf–angka–huruf"),
        ("Di atas ambang", f"{len(sure)}/{len(items)}", f"keyakinan ≥ {pct(thr, 0)}"),
        ("Rata-rata keyakinan", pct(sum(r.confidence for r in ok) / len(ok)) if ok else "–", "dari yang terbaca"),
    ])
    rows = []
    for name, _, r in items:
        p = r.parse
        status = ("Gagal" if not r.success else "Format tidak dikenali" if not p.valid
                  else "Perlu dicek" if r.confidence < thr else "Yakin")
        rows.append({"File": name, "Plat nomor": p.formatted if p else "", "Wilayah (perkiraan)": p.region if p else "",
                     "Keyakinan": r.confidence, "Koreksi karakter": p.corrections if p else 0, "Status": status})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Keyakinan": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    st.download_button("Unduh hasil (CSV)", table.to_csv(index=False).encode(), "hasil_plat_nomor.csv", "text/csv",
                       icon=":material/download:")

for name, data, r in items:
    st.write("")
    with st.container(key=f"card_res_{file_hash(data)}_{name}"):
        p = r.parse
        if not r.success:
            card_title(html.escape(name), pill("✕ Teks tidak terbaca", "bad"))
            st.image(data, width=360)
            st.caption("Coba ganti **Jenis gambar**, gunakan foto yang lebih tajam, atau crop area plat.")
            if r.error:
                st.code(r.error)
            continue
        badge = (pill("✓ Yakin", "ok") if p.valid and r.confidence >= thr else
                 pill("! Perlu dicek", "warn") if p.valid else pill("! Format tidak dikenali", "warn"))
        card_title(html.escape(name), badge)
        a, b = st.columns([1.3, 1], gap="medium")
        with a:
            st.image(r.viz_img, caption="Gambar input" + (" · kotak = area plat terdeteksi" if r.plate_found else ""),
                     width="stretch")
            if res["detect"] and r.plate_found is False:
                st.caption("Area plat tidak ditemukan otomatis, jadi seluruh gambar dibaca.")
        with b:
            dark = sum(r.plate_img.convert("L").getdata()) / (r.plate_img.width * r.plate_img.height) < 127
            st.markdown(f'<div class="plate{"" if dark else " white"}">{html.escape(p.formatted)}</div>',
                        unsafe_allow_html=True)
            stats([
                ("Keyakinan", pct(r.confidence), METHOD_LABEL.get(r.method, r.method)),
                ("Wilayah", p.region_code or "–", p.region or "–"),
                ("Teks mentah", p.raw or "–", f"{p.corrections} karakter dikoreksi" if p.corrections else "tanpa koreksi"),
            ])
            st.image(r.plate_img, caption="Area yang dibaca", width="stretch")
        with st.expander("Detail teknis"):
            st.markdown("**Percobaan per metode pra-pemrosesan**")
            st.dataframe(pd.DataFrame([{"Metode": METHOD_LABEL.get(k, k), "Teks": v.get("text", ""),
                                        "Keyakinan": v.get("confidence", 0.0), "Format valid": v.get("valid_format", False),
                                        "Galat": v.get("error", "")} for k, v in r.attempts.items()]),
                         hide_index=True, width="stretch")
            st.caption("Proses berhenti lebih awal bila hasil sudah valid dengan keyakinan ≥ 85%.")
            if r.tokens:
                st.markdown("**Potongan teks yang digabung (kiri → kanan)**")
                st.dataframe(pd.DataFrame([{"Teks": t["text"], "Keyakinan": round(t["conf"], 3),
                                            "Tinggi (px)": round(t["height"], 1)} for t in r.tokens]),
                             hide_index=True, width="stretch")
