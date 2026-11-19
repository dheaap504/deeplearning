"""Streamlit Deep Learning — aplikasi utama (multi-halaman).

Jalankan lokal:  streamlit run app.py
Di Streamlit Community Cloud, pilih file utama: app.py
"""
import streamlit as st

st.set_page_config(page_title="Streamlit Deep Learning", page_icon="🧠", layout="wide")

pages = [
    st.Page("halaman/beranda.py", title="Beranda", icon=":material/home:", default=True),
    st.Page("halaman/plat_nomor.py", title="Pengenalan Plat Nomor", icon=":material/directions_car:"),
    st.Page("halaman/klasifikasi_gambar.py", title="Klasifikasi Gambar (CNN)", icon=":material/image_search:"),
    st.Page("halaman/deteksi_karakter.py", title="Deteksi Karakter (Model Anda)", icon=":material/center_focus_strong:"),
]
st.navigation(pages).run()
