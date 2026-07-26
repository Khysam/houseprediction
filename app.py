"""
Aplikasi Streamlit — Prediksi & Rekomendasi Harga Rumah Tebet, Jakarta Selatan.

Menu:
0. Beranda           -> deskripsi lengkap website: apa yang bisa dilakukan, tentang
   dataset, tentang model, dan disclaimer.
1. Prediksi Harga     -> user isi spesifikasi rumah, model mengembalikan estimasi harga.
2. Rekomendasi Rumah  -> user isi budget & kriteria, aplikasi menampilkan rumah yang cocok
   dari dataset, lengkap dengan estimasi harga wajar model & indikator "good deal".
"""
import pathlib

import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config(
    page_title="Prediksi & Rekomendasi Rumah Tebet",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lokasi file — semua file (model, csv, txt) ada SEJAJAR dengan app.py
# di root repo, tidak di dalam sub-folder. BASE_DIR dipakai supaya app
# tetap bisa menemukan file ini walau Streamlit Cloud menjalankan app
# dari working directory yang berbeda.
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_prediksi_harga_rumah.pkl"
DATA_PATH = BASE_DIR / "data_rumah_clean.csv"
KELURAHAN_PATH = BASE_DIR / "daftar_kelurahan.txt"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"File model tidak ditemukan: `{MODEL_PATH.name}`. "
            "Pastikan file ini ada di root repo GitHub, sejajar dengan app.py."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            f"File data tidak ditemukan: `{DATA_PATH.name}`. "
            "Pastikan file ini ada di root repo GitHub, sejajar dengan app.py."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_daftar_kelurahan():
    if not KELURAHAN_PATH.exists():
        st.error(
            f"File daftar kelurahan tidak ditemukan: `{KELURAHAN_PATH.name}`. "
            "Pastikan file ini ada di root repo GitHub, sejajar dengan app.py."
        )
        st.stop()
    with open(KELURAHAN_PATH) as f:
        return [line.strip() for line in f if line.strip()]


model = load_model()
df = load_data()
daftar_kelurahan = load_daftar_kelurahan()

fitur_numerik = ["LB", "LT", "KT", "KM", "GRS"]


def format_rupiah(nilai):
    return f"Rp {nilai:,.0f}".replace(",", ".")


# ---------------------------------------------------------------------------
# Sidebar — pilihan menu
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 Menu")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["Beranda", "Prediksi Harga", "Rekomendasi Rumah"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df)} listing rumah kawasan Tebet, Jakarta Selatan "
    "(setelah dibersihkan dari duplikat & outlier ekstrem)."
)

# ===========================================================================
# MENU 0 — BERANDA / TENTANG APLIKASI
# ===========================================================================
if menu == "Beranda":
    st.title("🏠 Prediksi & Rekomendasi Harga Rumah Tebet, Jakarta Selatan")
    st.write(
        "Website ini membantu kamu memperkirakan **harga wajar rumah** di kawasan "
        "Tebet, Jakarta Selatan, sekaligus mencari **rekomendasi rumah** yang sesuai "
        "dengan budget dan kebutuhanmu — dibangun dari data listing rumah nyata di "
        "kawasan tersebut menggunakan model *machine learning*."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 Menu Prediksi Harga")
        st.write(
            "Masukkan spesifikasi rumah yang kamu maksud — luas bangunan, luas tanah, "
            "jumlah kamar tidur, kamar mandi, kapasitas garasi, dan kelurahan — lalu "
            "sistem akan memberi **estimasi harga wajar** berdasarkan pola harga rumah "
            "sejenis di dataset. Cocok dipakai untuk mengecek apakah harga jual/beli "
            "yang kamu dengar sudah wajar atau tidak, lengkap dengan daftar rumah "
            "pembanding dengan spesifikasi mirip."
        )
    with col2:
        st.subheader("🔍 Menu Rekomendasi Rumah")
        st.write(
            "Tentukan budget maksimal dan kriteria yang kamu inginkan (minimal kamar "
            "tidur/mandi, kapasitas garasi, kelurahan tertentu), sistem akan menyaring "
            "rumah dari dataset yang paling sesuai — dan menandai rumah mana yang "
            "**harga jualnya lebih murah dari estimasi wajar model** (potensi "
            "*good deal*), sehingga kamu bisa memprioritaskan rumah yang paling "
            "menguntungkan untuk ditindaklanjuti."
        )

    st.markdown("---")

    st.subheader("📊 Tentang Dataset")
    st.write(
        f"Model dilatih dari **{len(df)} data listing rumah** di kawasan Kecamatan "
        "Tebet, Jakarta Selatan, yang sudah dibersihkan dari data duplikat dan "
        "listing dengan harga ekstrem (outlier) supaya estimasi lebih realistis untuk "
        "rentang harga rumah pada umumnya di kawasan tersebut. Setiap listing memuat "
        "informasi:"
    )
    st.markdown(
        """
| Kolom | Keterangan |
|---|---|
| **Harga** | Harga jual rumah (Rp) |
| **LB** | Luas Bangunan (m²) |
| **LT** | Luas Tanah (m²) |
| **KT** | Jumlah Kamar Tidur |
| **KM** | Jumlah Kamar Mandi |
| **GRS** | Kapasitas garasi (jumlah mobil) |
| **Kelurahan** | Lokasi rumah (Tebet Timur, Tebet Barat, Kebon Baru, Bukit Duri, Menteng Dalam, atau "Tidak Diketahui" bila tidak disebutkan pada judul iklan) |
"""
    )

    st.subheader("⚙️ Tentang Model")
    st.write(
        "Estimasi harga dihasilkan oleh model **Decision Tree Regressor** yang dipilih "
        "setelah dibandingkan dengan 7 algoritma regresi lain (Linear Regression, "
        "Ridge, Lasso, Random Forest, Extra Trees, Gradient Boosting, dan "
        "K-Nearest Neighbors) — Decision Tree dipilih karena memberikan keseimbangan "
        "terbaik antara akurasi pada data uji dan ketahanan terhadap *overfitting*. "
        "Beberapa detail teknis:"
    )
    st.markdown(
        """
- Target harga dilatih dalam **skala logaritma** karena distribusi harga rumah
  sangat miring ke kanan (banyak rumah harga menengah, sedikit rumah harga sangat tinggi).
- Fitur numerik (`LB`, `LT`, `KT`, `KM`, `GRS`) di-*scaling* agar semua fitur
  diperlakukan adil oleh model.
- Fitur `Kelurahan` di-*encode* menggunakan One-Hot Encoding.
- Performa model pada data uji: **R² ≈ 0,73** — artinya model bisa menjelaskan
  sekitar 73% variasi harga rumah berdasarkan fitur-fitur di atas.
"""
    )

    st.info(
        "⚠️ **Disclaimer:** Estimasi harga pada website ini adalah **perkiraan "
        "statistik**, bukan penilaian resmi (appraisal). Faktor seperti kondisi "
        "fisik bangunan, legalitas/sertifikat, umur bangunan, arah hadap, dan "
        "kondisi pasar terkini tidak diperhitungkan oleh model. Gunakan hasil "
        "estimasi ini sebagai salah satu bahan pertimbangan, bukan acuan tunggal, "
        "dalam mengambil keputusan jual-beli rumah."
    )

    st.markdown("---")
    st.caption(
        "Dibuat menggunakan Python, scikit-learn, dan Streamlit. "
        "Silakan pilih menu di sidebar kiri untuk mulai menggunakan aplikasi."
    )

# ===========================================================================
# MENU 1 — PREDIKSI HARGA
# ===========================================================================
elif menu == "Prediksi Harga":
    st.title("🔮 Prediksi Harga Rumah")
    st.write(
        "Masukkan spesifikasi rumah di bawah ini untuk mendapatkan estimasi "
        "harga wajar berdasarkan model yang dilatih dari data listing rumah "
        "kawasan Tebet, Jakarta Selatan."
    )

    col1, col2 = st.columns(2)

    with col1:
        lb = st.number_input("Luas Bangunan (LB, m²)", min_value=10, max_value=2000, value=150, step=5)
        lt = st.number_input("Luas Tanah (LT, m²)", min_value=10, max_value=2000, value=120, step=5)
        kt = st.number_input("Jumlah Kamar Tidur (KT)", min_value=1, max_value=15, value=3, step=1)

    with col2:
        km = st.number_input("Jumlah Kamar Mandi (KM)", min_value=1, max_value=15, value=2, step=1)
        grs = st.number_input("Kapasitas Garasi / Carport (GRS)", min_value=0, max_value=10, value=1, step=1)
        kelurahan = st.selectbox(
            "Kelurahan",
            options=daftar_kelurahan,
            help=(
                "Pilih 'Tidak Diketahui' jika kelurahan spesifik tidak diketahui — "
                "model tetap bisa memberi estimasi berdasarkan luas & spesifikasi rumah."
            ),
        )

    if st.button("Prediksi Harga", type="primary"):
        input_df = pd.DataFrame([{
            "LB": lb, "LT": lt, "KT": kt, "KM": km, "GRS": grs, "KELURAHAN": kelurahan,
        }])
        pred = model.predict(input_df)[0]

        st.success(f"### Estimasi Harga: {format_rupiah(pred)}")
        st.caption(
            "Estimasi berdasarkan model Decision Tree yang dilatih dari data listing rumah "
            "kawasan Tebet. Angka ini adalah perkiraan wajar, bukan harga pasti — kondisi "
            "fisik bangunan, legalitas, dan faktor lain di luar data tidak diperhitungkan."
        )

        with st.expander("Lihat rumah pembanding dengan spesifikasi mirip"):
            mirip = df[
                (df["LT"].between(lt * 0.7, lt * 1.3))
                & (df["LB"].between(lb * 0.7, lb * 1.3))
            ].copy()
            if mirip.empty:
                st.write("Tidak ditemukan rumah pembanding dengan spesifikasi mirip di dataset.")
            else:
                mirip = mirip.sort_values("HARGA").head(10)
                st.dataframe(
                    mirip[["JUDUL", "KELURAHAN", "HARGA", "LB", "LT", "KT", "KM", "GRS"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ===========================================================================
# MENU 2 — REKOMENDASI RUMAH
# ===========================================================================
else:
    st.title("🔍 Rekomendasi Rumah Berdasarkan Kriteria")
    st.write(
        "Tentukan budget dan kriteria yang kamu inginkan, aplikasi akan menyaring rumah "
        "yang paling sesuai dari dataset — termasuk menandai rumah yang harganya "
        "**lebih murah dari estimasi wajar model** (kemungkinan *good deal*)."
    )

    with st.form("form_rekomendasi"):
        col1, col2 = st.columns(2)

        with col1:
            budget = st.number_input(
                "Budget Maksimal (Rp)", min_value=100_000_000,
                value=5_000_000_000, step=100_000_000, format="%d",
            )
            toleransi = st.slider(
                "Toleransi di bawah budget (%)", min_value=0, max_value=50, value=15,
                help="Rumah dengan harga sedikit di bawah budget juga akan ditampilkan.",
            ) / 100
            kt_min = st.number_input("Minimal Kamar Tidur", min_value=0, max_value=15, value=0, step=1)

        with col2:
            km_min = st.number_input("Minimal Kamar Mandi", min_value=0, max_value=15, value=0, step=1)
            grs_min = st.number_input("Minimal Kapasitas Garasi", min_value=0, max_value=10, value=0, step=1)
            kelurahan_filter = st.selectbox(
                "Kelurahan (opsional)",
                options=["Semua Kelurahan"] + daftar_kelurahan,
            )

        top_n = st.slider("Jumlah rekomendasi ditampilkan", min_value=3, max_value=30, value=10)

        submitted = st.form_submit_button("Cari Rekomendasi", type="primary")

    if submitted:
        hasil = df.copy()

        batas_bawah_budget = budget * (1 - toleransi)
        hasil = hasil[(hasil["HARGA"] >= batas_bawah_budget) & (hasil["HARGA"] <= budget)]

        if kt_min > 0:
            hasil = hasil[hasil["KT"] >= kt_min]
        if km_min > 0:
            hasil = hasil[hasil["KM"] >= km_min]
        if grs_min > 0:
            hasil = hasil[hasil["GRS"] >= grs_min]
        if kelurahan_filter != "Semua Kelurahan":
            hasil = hasil[hasil["KELURAHAN"] == kelurahan_filter]

        if hasil.empty:
            st.warning(
                "Tidak ada rumah yang sesuai dengan kriteria. "
                "Coba naikkan budget, kurangi kriteria, atau perbesar toleransi."
            )
        else:
            fitur_pred = hasil[["LB", "LT", "KT", "KM", "GRS", "KELURAHAN"]]
            hasil = hasil.copy()
            hasil["HARGA_PREDIKSI"] = model.predict(fitur_pred)
            hasil["SELISIH"] = hasil["HARGA"] - hasil["HARGA_PREDIKSI"]
            hasil = hasil.sort_values("SELISIH")

            st.success(f"Ditemukan {len(hasil)} rumah sesuai kriteria. Menampilkan top {min(top_n, len(hasil))}.")

            tampil = hasil.head(top_n).copy()
            tampil["Indikasi"] = tampil["SELISIH"].apply(
                lambda x: "💰 Lebih murah dari estimasi" if x < 0 else "Sesuai estimasi wajar"
            )

            st.dataframe(
                tampil[[
                    "JUDUL", "KELURAHAN", "HARGA", "HARGA_PREDIKSI", "SELISIH",
                    "LB", "LT", "KT", "KM", "GRS", "Indikasi",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "HARGA": st.column_config.NumberColumn("Harga", format="Rp %d"),
                    "HARGA_PREDIKSI": st.column_config.NumberColumn("Estimasi Wajar", format="Rp %d"),
                    "SELISIH": st.column_config.NumberColumn("Selisih", format="Rp %d"),
                },
            )
            st.caption(
                "Kolom **Selisih** negatif berarti harga jual rumah tersebut lebih rendah "
                "dari estimasi harga wajar model — berpotensi menjadi pilihan yang lebih hemat."
            )
