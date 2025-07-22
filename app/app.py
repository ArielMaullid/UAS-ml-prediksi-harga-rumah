import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import requests

# ================================
# KONFIGURASI LINK GOOGLE DRIVE
# ================================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1avasFCkdkthv6stzobl6iS83A5ou_umI"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1IzjsXM93ZN0dKQ2onHbdrGRNtAwMS8Tr"

def download_file(url, filename):
    try:
        if not os.path.exists(filename):
            response = requests.get(url)
            with open(filename, "wb") as f:
                f.write(response.content)
    except Exception as e:
        st.error(f"Gagal mengunduh {filename}: {e}")
        st.stop()

# Unduh file jika belum ada
download_file(MODEL_URL, "model.pkl")
download_file(SCALER_URL, "scaler.pkl")

# ================================
# LOAD MODEL & SCALER
# ================================
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error("❌ Gagal memuat model atau scaler. Periksa apakah file tersedia dan valid.")
    st.stop()

# ================================
# TAMPILAN STREAMLIT
# ================================
st.set_page_config(page_title="Prediksi Harga Rumah California", layout="centered")
st.title("🏠 Prediksi Harga Rumah California")
st.markdown("Masukkan fitur-fitur properti rumah berikut untuk memprediksi harga estimasinya.")

# ================================
# FORM INPUT USER
# ================================
with st.form("prediksi_form"):
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input("🌍 Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, format="%.4f")
        latitude = st.number_input("🌎 Latitude", min_value=32.0, max_value=42.0, value=37.0, format="%.4f")
        housing_age = st.number_input("🏚️ Usia Rumah (tahun)", min_value=1, max_value=100, value=30)
        median_income = st.number_input("💰 Pendapatan Median (10k USD)", min_value=0.5, max_value=15.0, value=4.0)
    with col2:
        total_rooms = st.number_input("🛏️ Total Kamar", min_value=1, max_value=50000, value=2000)
        total_bedrooms = st.number_input("🛏️ Total Kamar Tidur", min_value=1, max_value=10000, value=400)
        population = st.number_input("👥 Populasi", min_value=1, max_value=50000, value=1000)
        households = st.number_input("🏘️ Rumah Tangga", min_value=1, max_value=10000, value=350)

    submitted = st.form_submit_button("🔮 Prediksi Harga")

# ================================
# PROSES PREDIKSI
# ================================
if submitted:
    input_data = np.array([[longitude, latitude, housing_age, total_rooms,
                            total_bedrooms, population, households, median_income]])
    try:
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        st.success(f"💵 **Harga Rumah Diprediksi: ${prediction:,.2f}**")

        fig, ax = plt.subplots()
        ax.bar(["Prediksi Harga"], [prediction], color="skyblue")
        ax.set_ylabel("Harga ($)")
        ax.set_title("Hasil Prediksi")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
