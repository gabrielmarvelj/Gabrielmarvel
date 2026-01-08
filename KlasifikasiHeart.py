# ================================
# IMPORT LIBRARY
# ================================ 
# pandas, numpy: manipulasi data.
# matplotlib.pyplot, 
# seaborn: visualisasi data. joblib: menyimpan/memuat model.
# sklearn.*: machine learning, evaluasi model, dan tuning.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# ================================
# SETUP
# ================================ 
# Menampilkan semua kolom saat df.head(). 
# Menetapkan seed agar hasil random bisa direproduksi.
pd.set_option('display.max_columns', None)
np.random.seed(42)

# ================================
# 1. LOAD DATA DAN EDA
# ================================ 

print("1. Loading and EDA...")
# Daftar nama kolom sesuai dokumentasi dataset UCI
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'num'
]
# Membaca file .data dan mengubah "?" menjadi NaN
df = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")

# 👉 Tampilkan 5 baris awal data
# informasi tipe data (info()),
# jumlah missing values,
# dan data duplikat.

print("\n  Data Awal (5 baris):")
print(df.head())

print(f"Shape: {df.shape}")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

total_missing = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()
print(f"\n Total Missing Values: {total_missing}")
print(f" Duplikat Sebelum Dihapus: {duplicate_count}")

# ================================
# 2. VISUALISASI
# ================================
print("\n2. Visualisasi...")

#Histogram Semua Fitur Numerik
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig("numeric_histograms.png")
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()

# ================================
# 3. DATA CLEANING
# ================================
print("\n3. Data Cleaning...")

print("\n Missing value sebelum diperbaiki:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])  # Imputasi mode untuk kolom object
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())  # Imputasi median untuk numerik

# Tampilkan kembali missing value setelah diperbaiki
print("\n Missing value setelah diperbaiki:")
print(df.isnull().sum())

# Taampilkan data duplikat 
print(f"\n📑 Duplikat ditemukan: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    print("👉 Contoh data duplikat:")
    print(df[df.duplicated()].head())
# Hapus duplikat dan tampilkan jumlah yang dihapus
before_drop_duplicates = df.shape[0]
df.drop_duplicates(inplace=True)
after_drop_duplicates = df.shape[0]
removed_duplicates = before_drop_duplicates - after_drop_duplicates

print(f"\n Jumlah Data Duplikat yang Dihapus: {removed_duplicates}")
print(f"Data shape after cleaning: {df.shape}")
print("\n  Data Setelah Cleaning (5 baris):")
print(df.head())

# ================================
# 4. CORRELATION AFTER CLEANING
# ================================
print("\n4. Correlation after cleaning...")
# Visualisasi korelasi ulang setelah data dibersihkan
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, square=True, cmap="RdYlGn_r")
plt.title("Correlation After Cleaning")
plt.savefig("correlation_after_encoding.png")
plt.close()


# ================================
# 5. ENCODING & TARGET LABEL
# ================================
print("\n. Encoding target...")
# Ubah kolom 'num' jadi biner: 0 = tidak sakit, 1 = sakit
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
# Mapping hasil prediksi ke label
income_map = {0: "Tidak Sakit", 1: "Sakit"}
encoding_maps = {"num": income_map}

# Tampilkan 5 baris setelah encoding
print("\n  Data Setelah Encoding (5 baris):")
print(df.head())

# ================================
# 6. FEATURE SELECTION
# ================================

print("\n6. Feature Selection...")

# Fungsi untuk memilih fitur yang korelasinya > threshold
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_corr.add(corr_matrix.columns[i])
    return col_corr

# Pisahkan X (fitur) dan y (target)
X = df.drop("num", axis=1)
y = df["num"]

# Identifikasi fitur yang terlalu berkorelasi
corr_features = correlation(X, 0.8)
X.drop(columns=corr_features, inplace=True)

# Tampilkan nama fitur yang dipertahankan
print("\n  Fitur Setelah Seleksi Korelasi:")
print(X.columns.tolist())

# Tampilkan data fitur
print("\n  Data Fitur (5 baris):")
print(X.head())


# ================================
# 7. MODEL TRAINING
# ================================

print("\n7. Model Training...")

# Split data dengan proporsi 90:10 dan stratifikasi target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# ================================
# BASELINE: RANDOM FOREST DEFAULT
# ================================

print("\n📌 Random Forest Default")

rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

y_test_pred_default = rf_default.predict(X_test)

acc_default = accuracy_score(y_test, y_test_pred_default)
prec_default = precision_score(y_test, y_test_pred_default)
rec_default = recall_score(y_test, y_test_pred_default)

print("\nHasil Evaluasi Random Forest Default:")
print(f"Akurasi  : {acc_default:.4f}")
print(f"Precision: {prec_default:.4f}")
print(f"Recall   : {rec_default:.4f}")

# Grid pencarian parameter terbaik untuk RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

# Proses tuning model dengan GridSearchCV
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, scoring="accuracy", cv=3, n_jobs=-1, verbose=1)

try:
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")
except Exception as e:
    print("❌ GridSearchCV gagal:", str(e))
    exit()

# ================================
# 8. FEATURE IMPORTANCE
# ================================

# Visualisasi seberapa penting masing-masing fitur dalam model
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

# Barplot fitur penting
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("8. ✅ Feature Importance disimpan sebagai 'feature_importance.png'")


# ================================
# 9. PREDICTION TEST
# ================================

# Fungsi prediksi menggunakan model yang disimpan
def predict_heart_disease(data, model_components):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    model = model_components['model']
    features = model_components['feature_names']
    data_for_pred = data[features]
    pred = model.predict(data_for_pred)[0]
    prob = model.predict_proba(data_for_pred)[0]
    return {
        'prediction': pred,
        'prediction_label': model_components['target_map'][pred],
        'probability': prob[pred]
    }

# Uji prediksi pada satu sampel
test_sample = X_test.iloc[0].to_dict()
loaded = joblib.load("heart_disease_model.joblib")
print("\n Prediction test result:")
print(predict_heart_disease(test_sample, loaded))
print("Actual:", y_test.iloc[0])

# ================================
# 10. PROBABILITAS
# ================================

# Tampilkan contoh probabilitas prediksi
probas = best_model.predict_proba(X_test)
print("\n Contoh 5 Probabilitas Prediksi:")
print(probas[:5])


# ================================
# 11. Evaluasi Model
# ================================

# --- Evaluasi Data TRAINING ---
y_train_pred = best_model.predict(X_train)

print("\n Evaluasi Data Training:")
print(f"Akurasi  : {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall   : {recall_score(y_train, y_train_pred):.4f}")

# --- Evaluasi Data TEST ---
y_test_pred = best_model.predict(X_test)

print("\n Evaluasi Data Test:")
print(f"Akurasi  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_test_pred):.4f}")

# Confusion Matrix 
cm_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix")
print(cm_test)
# Visualisasi Confusion Matrix Test
plt.figure(figsize=(6, 4))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Sakit', 'Sakit'], yticklabels=['Tidak Sakit', 'Sakit'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()
print(" Confusion Matrix Test disimpan sebagai 'confusion_matrix.png'")

# ================================
# SAVE MODEL
# ================================

print("\n9. Saving model...")

# Simpan seluruh komponen model ke file joblib
model_components = {
    "model": best_model,
    "feature_names": X.columns.tolist(),
    "encoding_maps": encoding_maps,
    "model_params": grid.best_params_,
    "removed_features": list(corr_features),
    "target_map": income_map
}
joblib.dump(model_components, "heart_disease_model.joblib")
print("Model saved as 'heart_disease_model.joblib'")
