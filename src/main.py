import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ======================================
# LOAD MODEL & SCALER
# ======================================
try:
    model_path = os.path.join("models", "model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"❌ Gagal memuat model atau scaler. Detail error: {e}")
    st.stop()

# ======================================
# TITLE & DESKRIPSI
# ======================================
st.set_page_config(page_title="Prediksi Harga Rumah California", layout="centered")
st.title("🏠 Prediksi Harga Rumah California")
st.markdown("Masukkan fitur-fitur properti rumah berikut untuk memprediksi harga estimasinya.")

# ======================================
# INPUT FORM
# ======================================
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

# ======================================
# PREDIKSI & OUTPUT
# ======================================
if submitted:
    features = np.array([[longitude, latitude, housing_age, total_rooms,
                          total_bedrooms, population, households, median_income]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    st.success(f"💵 **Harga Rumah Diprediksi: ${prediction:,.2f}**")

    fig, ax = plt.subplots()
    ax.bar(["Harga Rumah"], [prediction], color="skyblue")
    ax.set_ylabel("Harga ($)")
    ax.set_title("Hasil Prediksi")
    st.pyplot(fig)
