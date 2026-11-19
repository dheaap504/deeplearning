"""Gaya visual & komponen kecil yang dipakai semua halaman."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "assets" / "contoh"
MODELS = ROOT / "models"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp button p, .stApp h1, .stApp h2, .stApp h3, .stApp td, .stApp th,
.stMarkdown { font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer { display: none !important; }
.hero h1 { font-size: 1.9rem !important; font-weight: 700; color: #111827; margin: 0 0 .2rem; letter-spacing: -.02em; }
.stMarkdown .hero p { color: #6B7280; font-size: .95rem !important; margin: 0; }
.eyebrow { font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #4F46E5; margin-bottom: .3rem; }
div[class*="st-key-card"] { background: #fff; border: 1px solid #ECEEF3; border-radius: 16px;
  padding: 1.1rem 1.2rem 1rem; box-shadow: 0 8px 24px rgba(17, 24, 39, .04); }
.stMarkdown p.card-t, .card-t { font-size: 1.05rem !important; font-weight: 700; color: #111827; margin: 0; }
.stMarkdown p.card-s, .card-s { font-size: .86rem !important; color: #6B7280; margin: .15rem 0 .6rem !important; line-height: 1.5; }
.plate { display: inline-block; background: #111; color: #fff; font-family: 'DejaVu Sans Mono', Consolas, monospace;
  font-weight: 700; font-size: 2rem; letter-spacing: .12em; padding: .45rem 1.2rem; border-radius: 10px;
  border: 3px solid #fff; outline: 3px solid #111; margin: .2rem 0 .6rem; }
.plate.white { background: #fff; color: #111; border-color: #111; outline-color: #fff; box-shadow: 0 0 0 1px #D1D5DB; }
.pill { display: inline-flex; align-items: center; gap: 4px; font-size: .75rem; font-weight: 700; padding: .18rem .6rem;
  border-radius: 999px; white-space: nowrap; }
.pill.ok { background: #E7F8EF; color: #0F7A44; } .pill.warn { background: #FFF4DB; color: #A15C00; }
.pill.bad { background: #FDECEC; color: #B42318; } .pill.info { background: #EEF0FF; color: #4338CA; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 150px), 1fr)); gap: .75rem; margin: .4rem 0 .8rem; }
.stat { background: #F8F9FC; border-radius: 12px; padding: .7rem .85rem; }
.stat .l { font-size: .75rem; color: #6B7280; font-weight: 600; }
.stat .v { font-size: 1.25rem; font-weight: 700; color: #111827; margin-top: 2px; overflow-wrap: anywhere; }
.stat .s { font-size: .72rem; color: #9CA3AF; margin-top: 2px; }
.tool { height: 100%; }
.tool .ic { width: 44px; height: 44px; border-radius: 12px; background: #EEF0FF; display: flex; align-items: center;
  justify-content: center; font-size: 22px; margin-bottom: .7rem; }
div[class*="st-key-row"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 1rem; }
div[class*="st-key-row"] [data-testid="stColumn"] { min-width: min(100%, 300px); }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card"]) { flex: 1 1 auto; }
[data-testid="stColumn"] div[class*="st-key-card"] { height: 100%; }
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, desc: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{desc}</p></div>',
                unsafe_allow_html=True)
    st.write("")


def card_title(title: str, sub: str = ""):
    st.markdown(f'<p class="card-t">{title}</p>' + (f'<p class="card-s">{sub}</p>' if sub else ""), unsafe_allow_html=True)


def stats(items):
    st.markdown('<div class="stats">' + "".join(
        f'<div class="stat"><div class="l">{l}</div><div class="v">{v}</div>' + (f'<div class="s">{s}</div>' if s else "")
        + "</div>" for l, v, s in items) + "</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def pct(v: float, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%".replace(".", ",")
