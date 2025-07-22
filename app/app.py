import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import requests

# ======================================
# LINK GOOGLE DRIVE (gunakan export=download)
# ======================================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1avasFCkdkthv6stzobl6iS83A5ou_umI"
SCALER_URL = "https://drive.google.com/uc?export=download&id=1IzjsXM93ZN0dKQ2onHbdrGRNtAwMS8Tr"

# ======================================
# FUNGSI UNDUH FILE
# ======================================
def download_file(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
    except Exception as e:
        st.error(f"Gagal mengunduh {filename}. Error: {e}")
        st.stop()

# ======================================
# CEK & UNDUH FILE MODEL
# ======================================
if not os.path.exists("model.pkl"):
    download_file(MODEL_URL, "model.pkl")
if not os.path.exists("scaler.pkl"):
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
    st.error("❌ Gagal memuat model atau scaler. Periksa apakah file tersedia.")
    st.stop()

# ======================================
# TITLE & FORM
# ======================================
st.set_page_config(page_title="Prediksi Harga Rumah California", layout="centered")
st.title("🏠 Prediksi Harga Rumah California")
st.markdown("Masukkan fitur-fitur rumah untuk memprediksi harga estimasi:")

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

    submit = st.form_submit_button("🔮 Prediksi")

# ======================================
# PREDIKSI
# ======================================
if submit:
    features = np.array([[longitude, latitude, housing_age, total_rooms,
                          total_bedrooms, population, households, median_income]])
    scaled = scaler.transform(features)
    result = model.predict(scaled)[0]

    st.success(f"💵 **Harga Rumah Diprediksi: ${result:,.2f}**")

    # Visualisasi prediksi
    fig, ax = plt.subplots()
    ax.bar(["Harga Prediksi"], [result], color="skyblue")
    ax.set_ylabel("Harga ($)")
    ax.set_title("Visualisasi Prediksi")
    st.pyplot(fig)
