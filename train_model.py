"""
Script untuk melatih ulang model prediksi harga rumah Tebet
dan menyimpan (1) pipeline model & (2) dataset bersih yang dipakai
oleh aplikasi Streamlit (app.py).

Jalankan sekali di lokal (butuh DATA_RUMAH.xlsx di folder yang sama)
untuk menghasilkan:
- model/model_prediksi_harga_rumah.pkl
- data/data_rumah_clean.csv
"""
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

import joblib

# 1. Load data ----------------------------------------------------------
df = pd.read_excel('DATA_RUMAH.xlsx')

kolom_numerik_inti = ['LB', 'LT', 'KT', 'KM']
df = df.dropna(subset=kolom_numerik_inti).reset_index(drop=True)
if 'GRS' in df.columns:
    df['GRS'] = df['GRS'].fillna(0)

df = df.drop_duplicates(
    subset=['NAMA RUMAH', 'HARGA', 'LB', 'LT', 'KT', 'KM', 'GRS']
).reset_index(drop=True)

# 2. Feature engineering: parsing NAMA RUMAH ----------------------------
daftar_kelurahan = [
    'Tebet Timur', 'Tebet Barat', 'Kebon Baru',
    'Bukit Duri', 'Manggarai Selatan', 'Manggarai', 'Menteng Dalam'
]

def ekstrak_kelurahan(teks):
    teks_lower = str(teks).lower()
    for kel in daftar_kelurahan:
        pattern = r'\b' + re.escape(kel.lower()) + r'\b'
        if re.search(pattern, teks_lower):
            return kel
    return 'Tidak Diketahui'

def ekstrak_judul(teks):
    return str(teks).split(',')[0].strip()

df['JUDUL'] = df['NAMA RUMAH'].apply(ekstrak_judul)
df['KELURAHAN'] = df['NAMA RUMAH'].apply(ekstrak_kelurahan)

# 3. Buang outlier HARGA ekstrem (1.5xIQR) -------------------------------
Q1 = df['HARGA'].quantile(0.25)
Q3 = df['HARGA'].quantile(0.75)
IQR = Q3 - Q1
batas_bawah = max(Q1 - 1.5 * IQR, 0)
batas_atas = Q3 + 1.5 * IQR
df_model = df[(df['HARGA'] >= batas_bawah) & (df['HARGA'] <= batas_atas)].copy()

# 4. Preprocessing & training --------------------------------------------
fitur_numerik = ['LB', 'LT', 'KT', 'KM', 'GRS']
fitur_kategorikal = ['KELURAHAN']

X = df_model[fitur_numerik + fitur_kategorikal]
y = df_model['HARGA']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

try:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('kategorikal', encoder, fitur_kategorikal),
        ('numerik', StandardScaler(), fitur_numerik),
    ]
)

model_ttr = TransformedTargetRegressor(
    regressor=DecisionTreeRegressor(max_depth=6, random_state=42),
    func=np.log1p, inverse_func=np.expm1
)

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_ttr)])
pipe.fit(X_train, y_train)

r2_test = r2_score(y_test, pipe.predict(X_test))
print(f'R2 pada data uji: {r2_test:.3f}')

# 5. Simpan model & dataset bersih ---------------------------------------
joblib.dump(pipe, 'model/model_prediksi_harga_rumah.pkl')

kolom_simpan = ['JUDUL', 'KELURAHAN', 'HARGA', 'LB', 'LT', 'KT', 'KM', 'GRS']
df_model[kolom_simpan].to_csv('data/data_rumah_clean.csv', index=False)

# simpan juga daftar kelurahan yang valid, untuk dropdown di aplikasi
daftar_kelurahan_valid = sorted(df_model['KELURAHAN'].unique().tolist())
with open('data/daftar_kelurahan.txt', 'w') as f:
    f.write('\n'.join(daftar_kelurahan_valid))

print('Selesai. Model dan data bersih tersimpan di folder model/ dan data/.')
