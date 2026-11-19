"""Halaman: klasifikasi gambar dengan CNN kustom (CIFAR-10)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.cnn_model import CLASSES, CLASSES_ID
from src.cnn_tools import load_from_bytes, load_metrics, load_pretrained, predict, to_bytes, train
from src.ui import SAMPLES, card_title, hero, pct, pill, setup, stats

setup()
hero("PyTorch · CNN kustom", "Klasifikasi Gambar",
     "CNN 3 blok konvolusi yang dilatih dari nol pada CIFAR-10 untuk mengenali 10 jenis objek.")

INDIGO, LIGHT, GRID = "#4F46E5", "#C7D2FE", "#F0F1F5"
LABELS = [f"{CLASSES_ID[c]} ({c})" for c in CLASSES]


def base(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=8, r=16, t=10, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
                      showlegend=False)
    fig.update_xaxes(showgrid=False, automargin=True, fixedrange=True)
    fig.update_yaxes(gridcolor=GRID, automargin=True, fixedrange=True)
    return fig


@st.cache_resource
def pretrained():
    return load_pretrained()


if "cnn_model" not in st.session_state:
    m, ok = pretrained()
    st.session_state.cnn_model = m
    st.session_state.cnn_source = "pralatih" if ok else "acak"
src = st.session_state.cnn_source
src_label = {"pralatih": pill("✓ Bobot pralatih (repo)", "ok"), "acak": pill("! Bobot acak — belum dilatih", "bad"),
             "latih": pill("✓ Hasil latih ulang sesi ini", "info"), "unggah": pill("✓ Bobot dari file Anda", "info")}[src]
st.markdown(f"Model yang sedang dipakai: {src_label}", unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs([":material/image_search: Prediksi", ":material/analytics: Evaluasi model",
                            ":material/model_training: Latih ulang & bobot"])

# ------------------------------------------------------------------ prediksi
with tab1:
    with st.container(key="row_pred"):
        c1, c2 = st.columns([1, 1.4], gap="medium")
        with c1, st.container(key="card_pin"):
            card_title("Gambar", "unggah foto, atau pilih salah satu contoh dari data uji CIFAR-10")
            up = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
            pick = st.selectbox("…atau pilih contoh", ["—"] + LABELS, index=4, disabled=up is not None)
            img = None
            if up:
                img = Image.open(up).convert("RGB")
            elif pick != "—":
                img = Image.open(SAMPLES / f"cifar_{CLASSES[LABELS.index(pick)]}.jpg").convert("RGB")
            if img:
                st.image(img.resize((224, 224), Image.NEAREST) if max(img.size) < 100 else img, width=260,
                         caption=f"{img.size[0]}×{img.size[1]} px → diperkecil ke 32×32 untuk model")
            st.caption("Model ini paling cocok untuk foto satu objek di tengah gambar, dari 10 kelas: "
                       + ", ".join(CLASSES_ID.values()) + ".")
        with c2, st.container(key="card_pout"):
            card_title("Hasil prediksi")
            if img is None:
                st.info("Unggah gambar atau pilih contoh.", icon=":material/info:")
            else:
                prob = predict(st.session_state.cnn_model, img)
                order = np.argsort(prob)[::-1]
                top = order[0]
                stats([
                    ("Prediksi", CLASSES_ID[CLASSES[top]], CLASSES[top]),
                    ("Keyakinan", pct(prob[top]), "probabilitas softmax"),
                    ("Pilihan kedua", CLASSES_ID[CLASSES[order[1]]], pct(prob[order[1]])),
                ])
                if src == "acak":
                    st.warning("Model belum dilatih, sehingga prediksi ini acak.", icon=":material/warning:")
                elif prob[top] < 0.5:
                    st.caption("Keyakinan rendah: gambar mungkin bukan salah satu dari 10 kelas, atau objeknya ambigu.")
                idx = order[::-1]
                fig = go.Figure(go.Bar(x=prob[idx] * 100, y=[LABELS[i] for i in idx], orientation="h",
                                       marker=dict(color=[INDIGO if i == top else LIGHT for i in idx], cornerradius=5),
                                       text=[f"{p * 100:.1f}%".replace(".", ",") for p in prob[idx]],
                                       textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
                fig = base(fig, 360)
                fig.update_xaxes(title="Probabilitas (%)", range=[0, 112], ticksuffix="%")
                fig.update_yaxes(title="Kelas", showgrid=False)
                st.plotly_chart(fig, config={"displayModeBar": False})

# ------------------------------------------------------------------ evaluasi
with tab2:
    met = load_metrics()
    if not met:
        st.info("Belum ada metrik evaluasi (file models/cnn_cifar10_metrics.json tidak ditemukan).")
    else:
        with st.container(key="card_ev"):
            card_title("Kinerja bobot pralatih pada data uji", f"{met['n_test']:,} gambar uji CIFAR-10 yang tidak dipakai saat latih".replace(",", "."))
            pc = met["per_class"]
            best, worst = max(pc, key=pc.get), min(pc, key=pc.get)
            stats([
                ("Akurasi uji", f"{met['test_accuracy']:.1f}%".replace(".", ","), f"tebakan acak = 10%"),
                ("Epoch", met["epochs"], f"{met['n_train']:,} gambar latih".replace(",", ".")),
                ("Kelas terbaik", CLASSES_ID[best], f"{pc[best]:.1f}%".replace(".", ",")),
                ("Kelas tersulit", CLASSES_ID[worst], f"{pc[worst]:.1f}%".replace(".", ",")),
            ])
        st.write("")
        with st.container(key="row_ev"):
            c1, c2 = st.columns(2, gap="medium")
            with c1, st.container(key="card_pc"):
                card_title("Akurasi per kelas", "persentase gambar uji tiap kelas yang ditebak benar")
                s = pd.Series({CLASSES_ID[k]: v for k, v in pc.items()}).sort_values()
                fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                       marker=dict(color=INDIGO, cornerradius=5),
                                       text=[f"{v:.1f}%".replace(".", ",") for v in s.values], textposition="outside",
                                       cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
                fig = base(fig, 360)
                fig.update_xaxes(title="Akurasi (%)", range=[0, 112], ticksuffix="%")
                fig.update_yaxes(title="Kelas", showgrid=False)
                st.plotly_chart(fig, config={"displayModeBar": False})
            with c2, st.container(key="card_hist"):
                card_title("Kurva pelatihan", "akurasi data latih vs data uji per epoch")
                h = pd.DataFrame(met["history"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=h.epoch, y=h.train_acc, name="Data latih", mode="lines+markers",
                                         line=dict(color=LIGHT, width=3)))
                fig.add_trace(go.Scatter(x=h.epoch, y=h.test_acc, name="Data uji", mode="lines+markers",
                                         line=dict(color=INDIGO, width=3)))
                fig = base(fig, 360)
                fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
                fig.update_xaxes(title="Epoch", dtick=2)
                fig.update_yaxes(title="Akurasi (%)", ticksuffix="%")
                st.plotly_chart(fig, config={"displayModeBar": False})
        st.write("")
        with st.container(key="card_cm"):
            card_title("Matriks kebingungan (confusion matrix)",
                       "baris = kelas sebenarnya, kolom = tebakan model; diagonal = benar. Kesalahan terbanyak biasanya "
                       "antara kucing ↔ anjing dan mobil ↔ truk.")
            cm = np.array(met["confusion"])
            names = [CLASSES_ID[c] for c in CLASSES]
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, colorscale=[[0, "#F8F9FC"], [0.15, "#C7D2FE"], [1, "#312E81"]],
                                       text=cm, texttemplate="%{text}", xgap=2, ygap=2,
                                       hovertemplate="Sebenarnya %{y}, ditebak %{x}: %{z} gambar<extra></extra>",
                                       colorbar=dict(title="Jumlah", thickness=10, outlinewidth=0)))
            fig = base(fig, 470)
            fig.update_xaxes(title="Tebakan model", side="bottom")
            fig.update_yaxes(title="Kelas sebenarnya", autorange="reversed", showgrid=False)
            st.plotly_chart(fig, config={"displayModeBar": False})

# ------------------------------------------------------------------ latih ulang & bobot
with tab3:
    with st.container(key="row_tr"):
        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1, st.container(key="card_train"):
            card_title("Latih ulang di server", "melatih dari nol pada CIFAR-10 — hasil hanya berlaku untuk sesi ini")
            st.warning("Pelatihan berjalan di CPU: ±1–4 menit per epoch dan mengunduh dataset ±170 MB saat pertama kali. "
                       "Di Streamlit Cloud gratis, gunakan 1–3 epoch saja.", icon=":material/schedule:")
            a, b, c = st.columns(3)
            ep = a.number_input("Epoch", 1, 30, 2)
            bs = b.selectbox("Ukuran batch", [32, 64, 128], index=1)
            lr = c.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            if st.button("Mulai latih", type="primary", icon=":material/play_arrow:"):
                from src.cnn_model import CNNClassifier
                model = CNNClassifier(len(CLASSES))
                bar = st.progress(0.0, text="Menyiapkan dataset…")
                hist = train(model, int(ep), int(bs), float(lr), on_progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
                bar.empty()
                st.session_state.cnn_model, st.session_state.cnn_source = model, "latih"
                st.session_state.cnn_hist = hist
                st.success(f"Selesai. Akurasi uji: {hist[-1]['test_acc']:.1f}%".replace(".", ","))
            if st.session_state.get("cnn_hist"):
                st.dataframe(pd.DataFrame(st.session_state.cnn_hist).rename(columns={
                    "epoch": "Epoch", "loss": "Loss", "train_acc": "Akurasi latih (%)", "test_acc": "Akurasi uji (%)"}),
                    hide_index=True, width="stretch")
        with c2, st.container(key="card_w"):
            card_title("Simpan & muat bobot", "agar hasil latih tidak hilang saat aplikasi dimulai ulang")
            st.download_button("Unduh bobot model aktif (.pth)", to_bytes(st.session_state.cnn_model),
                               "cnn_cifar10.pth", icon=":material/download:", width="stretch")
            wf = st.file_uploader("Muat bobot (.pth) dengan arsitektur yang sama", type=["pth", "pt"])
            if wf and st.button("Pakai bobot ini", icon=":material/upload:", width="stretch"):
                try:
                    st.session_state.cnn_model = load_from_bytes(wf.getvalue())
                    st.session_state.cnn_source = "unggah"
                    st.rerun()
                except Exception as e:
                    st.error(f"Bobot tidak cocok dengan arsitektur CNN ini: {e}")
            if st.session_state.cnn_source != "pralatih" and pretrained()[1]:
                if st.button("Kembali ke bobot pralatih", width="stretch"):
                    st.session_state.cnn_model, st.session_state.cnn_source = pretrained()[0], "pralatih"
                    st.rerun()
            st.caption("Agar bobot baru menjadi bawaan aplikasi, ganti file `models/cnn_cifar10.pth` di repo GitHub.")
