# 🏠 Prediksi & Rekomendasi Harga Rumah Tebet, Jakarta Selatan

Aplikasi Streamlit untuk memprediksi harga rumah dan mencari rekomendasi rumah
berdasarkan budget & kriteria, dilatih dari data listing rumah kawasan Tebet,
Jakarta Selatan.

## Fitur

1. **Menu Prediksi Harga** — masukkan spesifikasi rumah (luas bangunan, luas
   tanah, kamar tidur, kamar mandi, garasi, kelurahan) untuk mendapat estimasi
   harga wajar dari model.
2. **Menu Rekomendasi Rumah** — masukkan budget & kriteria minimal (kamar
   tidur/mandi/garasi/kelurahan), aplikasi menyaring rumah dari dataset dan
   menandai mana yang harganya lebih murah dari estimasi wajar model
   (potensi *good deal*).

## Struktur folder

```
.
├── app.py                          # aplikasi Streamlit (2 menu)
├── train_model.py                  # script untuk melatih ulang model (opsional)
├── requirements.txt
├── model/
│   └── model_prediksi_harga_rumah.pkl   # pipeline model terlatih (sudah tersedia)
└── data/
    ├── data_rumah_clean.csv        # dataset bersih hasil feature engineering
    └── daftar_kelurahan.txt        # daftar kelurahan valid untuk dropdown
```

Model (`.pkl`) dan dataset bersih (`.csv`) di folder `model/` dan `data/`
**sudah disertakan dan siap pakai** — kamu tidak perlu melatih ulang untuk
menjalankan aplikasi.

## Menjalankan di lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

## Upload ke GitHub

```bash
git init
git add .
git commit -m "Aplikasi prediksi & rekomendasi harga rumah Tebet"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

> Catatan: file data mentah `DATA_RUMAH.xlsx` sengaja **tidak** disertakan di
> `.gitignore` default repo ini karena app hanya butuh `model/*.pkl` dan
> `data/*.csv` yang sudah jadi. Kalau kamu ingin bisa melatih ulang model dari
> GitHub Actions atau di komputer lain, taruh `DATA_RUMAH.xlsx` di root folder
> lalu hapus baris `DATA_RUMAH.xlsx` di `.gitignore`.

## Deploy ke Streamlit Community Cloud

1. Pastikan repo GitHub di atas sudah **public** (atau private dengan akun
   Streamlit yang terhubung ke GitHub-mu).
2. Buka **[share.streamlit.io](https://share.streamlit.io)** dan login dengan
   akun GitHub.
3. Klik **"New app"**.
4. Pilih repo, branch (`main`), dan **Main file path**: `app.py`.
5. Klik **"Deploy"**. Streamlit Cloud akan otomatis membaca `requirements.txt`
   dan menginstal dependensi.
6. Setelah build selesai (1-3 menit), aplikasi akan tersedia di URL publik
   seperti `https://<nama-app>.streamlit.app`.

Jika ke depannya kamu update model (`model/model_prediksi_harga_rumah.pkl`)
atau data, cukup `git push` lagi — Streamlit Cloud otomatis me-redeploy versi
terbaru.

## Melatih ulang model (opsional)

Kalau ada data baru atau ingin mengganti algoritma:

```bash
# taruh DATA_RUMAH.xlsx di root folder ini, lalu:
python train_model.py
```

Script ini akan menimpa ulang `model/model_prediksi_harga_rumah.pkl`,
`data/data_rumah_clean.csv`, dan `data/daftar_kelurahan.txt`.

## Catatan teknis

- Model: Decision Tree Regressor (`max_depth=6`), dibungkus
  `TransformedTargetRegressor` (target `HARGA` dilatih dalam skala log karena
  distribusinya right-skewed), dengan `StandardScaler` untuk fitur numerik
  dan `OneHotEncoder` untuk `KELURAHAN`.
- R² pada data uji: ±0.73 (bisa bervariasi tergantung random split saat
  retraining).
- Kolom `KELURAHAN` untuk sebagian besar listing bernilai `"Tidak Diketahui"`
  karena judul iklan sumber data sering tidak menyebut nama kelurahan secara
  eksplisit — ini keterbatasan data, bukan bug parsing. Model tidak terlalu
  bergantung pada fitur ini (kontribusi kelurahan pada feature importance
  sangat kecil dibanding `LT`/`LB`).
