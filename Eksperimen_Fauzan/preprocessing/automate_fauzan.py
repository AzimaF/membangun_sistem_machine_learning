import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# === 1. Load Dataset ===
df = pd.read_csv(r"C:\Users\whand\Downloads\membangun_sistem_machine_learning-main\membangun_sistem_machine_learning-main\Eksperimen_Fauzan\crop_recom_raw\Crop_recommendation.csv")

# === 2. Tangani Missing Values ===
missing_before = df.isnull().sum()
print("🔍 Missing values sebelum diproses:\n", missing_before)
df = df.dropna()  # Jika ingin mengisi bisa pakai df.fillna(method='ffill') misalnya

# === 3. Hapus Duplikat ===
duplicates_before = df.duplicated().sum()
print(f"\n🧹 Duplikasi ditemukan: {duplicates_before} baris")
df = df.drop_duplicates()

# === 4. Deteksi dan Tangani Outlier (menggunakan IQR) ===
def remove_outliers_iqr(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = remove_outliers_iqr(df, numerical_cols)

# === 5. Pisahkan Fitur dan Target ===
X = df.drop("label", axis=1)
y = df["label"]

# === 6. Standardisasi ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 7. Simpan Data yang Telah Diproses ===
df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
df_preprocessed["label"] = y.values

df_preprocessed.to_csv("membangun_sistem_machine_learning-main\Eksperimen_Fauzan\preprocessing\data_preprocessed.csv", index=False)
print("\n✅ Data hasil preprocessing berhasil disimpan ke 'data_preprocessed.csv'")