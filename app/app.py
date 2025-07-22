import streamlit as st
import numpy as np
import pickle
import requests
import os
import matplotlib.pyplot as plt

# ======================================
# LINK GOOGLE DRIVE (direct download)
# ======================================
MODEL_URL = "https://drive.google.com/uc?id=1oCqaHeXjcvyEQRenCSa91D9zWqxWTkUt"
SCALER_URL = "https://drive.google.com/uc?id=1Dm30dB3oS4luSQRZNlGkYjHKWn16i-rw"

# ======================================
# DOWNLOAD FILE MODEL & SCALER
# ======================================
def download_file(url, filename):
    if not os.path.exists(filename):
        with requests.get(url) as r:
            if r.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(r.content)
            else:
                st.error(f"Gagal mengunduh {filename}")
                st.stop()

download_file(MODEL_URL, "model.pkl")
download_file(SCALER_URL, "scaler.pkl")

# ======================================
# LOAD MODEL & SCALER
# ======================================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error("❌ Gagal memuat model atau scaler. Periksa apakah file tersedia dan valid.")
    st.stop()

# ======================================
# TITLE & DESKRIPSI
# ======================================
st.set_page_config(page_title="Prediksi Harga Rumah California", layout="centered")
st.title("🏠 Prediksi Harga Rumah California")
st.markdown("Masukkan fitur-fitur properti berikut untuk memprediksi harga rumah.")

# ======================================
# INPUT FORM
# ======================================
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("🌍 Longitude", -125.0, -114.0, -120.0, format="%.4f")
        latitude = st.number_input("🌎 Latitude", 32.0, 42.0, 37.0, format="%.4f")
        housing_age = st.number_input("🏚️ Usia Rumah (tahun)", 1, 100, 30)
        median_income = st.number_input("💰 Pendapatan Median (10k USD)", 0.5, 15.0, 4.0)

    with col2:
        total_rooms = st.number_input("🛏️ Total Kamar", 1, 50000, 2000)
        total_bedrooms = st.number_input("🛏️ Total Kamar Tidur", 1, 10000, 400)
        population = st.number_input("👥 Populasi", 1, 50000, 1000)
        households = st.number_input("🏘️ Rumah Tangga", 1, 10000, 350)

    submitted = st.form_submit_button("🔮 Prediksi Harga")

# ======================================
# PREDIKSI & OUTPUT
# ======================================
if submitted:
    data = np.array([[longitude, latitude, housing_age, total_rooms,
                      total_bedrooms, population, households, median_income]])

    scaled_data = scaler.transform(data)
    prediksi = model.predict(scaled_data)[0]

    st.success(f"💵 **Harga Rumah Diprediksi: ${prediksi:,.2f}**")

    fig, ax = plt.subplots()
    ax.bar(["Harga Rumah"], [prediksi], color="skyblue")
    ax.set_ylabel("Harga ($)")
    ax.set_title("Hasil Prediksi")
    st.pyplot(fig)
