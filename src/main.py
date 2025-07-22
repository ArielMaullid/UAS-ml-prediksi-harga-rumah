import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# STEP 1: Load Dataset
# ============================================
print("📥 Loading dataset...")
df = pd.read_csv('./data/housing.csv')

# Hapus baris yang mengandung nilai kosong
df.dropna(inplace=True)

# Hapus kolom kategorikal jika ada
if "ocean_proximity" in df.columns:
    print("🧹 Menghapus kolom 'ocean_proximity' (kategorikal)...")
    df = df.drop("ocean_proximity", axis=1)

# ============================================
# STEP 2: Pisahkan Fitur dan Target
# ============================================
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ============================================
# STEP 3: Split dan Scaling Data
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 4: Train Model
# ============================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ============================================
# STEP 5: Simpan Model
# ============================================
os.makedirs('./models', exist_ok=True)

with open('./models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('./models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ============================================
# STEP 6: Evaluasi Model
# ============================================
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model evaluation:")
print(f"   - MSE  : {mse:,.2f}")
print(f"   - RMSE : {rmse:,.2f}")
print(f"   - R2   : {r2:.4f}")
