# ===============================
# IMPORT LIBRARY
# ===============================
import streamlit as st  # Streamlit untuk membangun web app interaktif
import pandas as pd     # Pandas untuk manipulasi data
import numpy as np      # Numpy untuk operasi numerik
import joblib           # Joblib untuk memuat model Machine Learning
import plotly.express as px  # Plotly Express untuk visualisasi grafik
import plotly.graph_objects as go  # Plotly GO untuk grafik tingkat lanjut (gauge chart)
from datetime import datetime  # Untuk menangani waktu dan tanggal
import json             # Untuk ekspor hasil prediksi ke format JSON

# ===============================
# PAGE SETUP
# =============================== 
# Konfigurasi tampilan halaman aplikasi Streamlit
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",  # Judul tab browser
    page_icon="💓",                         # Ikon tab
    layout="wide",                          # Layout halaman lebar
    initial_sidebar_state="collapsed"       # Sidebar tidak ditampilkan awalnya
)

# Judul utama dan deskripsi aplikasi
st.title("💓 Prediksi Penyakit Jantung")
st.markdown("Aplikasi prediksi risiko penyakit jantung berdasarkan data " \
"medis pasien menggunakan model **Random Forest**.")

# ===============================
# LOAD MODEL
# ===============================
# Fungsi untuk memuat model dari file joblib
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.joblib")  # Memuat file model yang berisi model dan metadata

# Memanggil fungsi untuk memuat model
model_bundle = load_model()
model = model_bundle["model"]              # Model Machine Learning
features = model_bundle["feature_names"]   # Fitur yang digunakan dalam model
target_map = model_bundle["target_map"]    # Pemetaan hasil prediksi ke label kelas

# ===============================
# INPUT MAPPING
# ===============================
# Mapping nilai dari input form (teks) ke angka yang dipahami model
sex_map = {"Laki-laki": 1, "Perempuan": 0}
cp_map = {"Nyeri saat aktivitas berat": 1, "Nyeri tidak khas": 2, "Nyeri bukan karena jantung": 3, "Tidak ada gejala": 4}
fbs_map = {"Ya (>120 mg/dl)": 1, "Tidak (<=120 mg/dl)": 0}
restecg_map = {"Normal": 0, "ST-T Abnormalitas": 1, "Hypertrophy(pembesaran jantung)": 2}
exang_map = {"Ya": 1, "Tidak": 0}
slope_map = {"Upsloping(meningkat, normal)": 1, "Flat(datar, bisa berisiko)": 2, "Downsloping(menurun, berisiko tinggi)": 3}
thal_map = {"Normal": 3, "Fixed Defect(kerusakan permanen)": 6, "Reversible Defect(kerusakan yang bisa pulih)": 7}

# ===============================
# FORM INPUT
# ===============================
# Bagian tampilan form input data pasien
col_form, col_result = st.columns([2, 1])  # Layout dua kolom

with col_form:
    st.subheader("📝 Input Data Pasien")
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            # Input kolom kiri
            age = st.number_input("Umur(age)", min_value=20, max_value=80, value=st.session_state.get("age", 50), key="age")
            sex = st.selectbox("Jenis Kelamin(sex)", list(sex_map.keys()), key="sex")
            cp = st.selectbox("Tipe Nyeri Dada(cp)", list(cp_map.keys()), key="cp")
            trestbps = st.number_input("Tekanan Darah Saat Istirahat (mm Hg)(trestbps)", min_value=80, max_value=200, value=st.session_state.get("trestbps", 120), key="trestbps")
            chol = st.number_input("Kolesterol (mg/dl)(chol)", min_value=100, max_value=600, value=st.session_state.get("chol", 240), key="chol")
            fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?(fbs)", list(fbs_map.keys()), key="fbs")
            restecg = st.selectbox("Hasil Elektrokardiografi Saat Istirahat(restecg)", list(restecg_map.keys()), key="restecg")
        with col2:
            # Input kolom kanan
            thalach = st.number_input("Detak Jantung Maksimum(thalach)", min_value=60, max_value=220, value=st.session_state.get("thalach", 150), key="thalach")
            exang = st.selectbox("Angina(Nyeridada) akibat Olahraga?(exang)", list(exang_map.keys()), key="exang")
            oldpeak = st.number_input("Oldpeak (Depresi ST)(oldpeak)", min_value=0.0, max_value=6.0, value=st.session_state.get("oldpeak", 1.0), key="oldpeak")
            slope = st.selectbox("Kemiringan ST(slope)", list(slope_map.keys()), key="slope")
            ca = st.number_input("Jumlah Pembuluh Darah Utama (0-3)(ca)", min_value=0, max_value=3, value=st.session_state.get("ca", 0), key="ca")
            thal = st.selectbox("Tipe Thalassemia(thal)", list(thal_map.keys()), key="thal")

        # Tombol submit, reset, dan export
        col_submit, col_reset, col_export = st.columns([1, 1, 1])
        submit_btn = col_submit.form_submit_button("🔍 Prediksi")
        reset_btn = col_reset.form_submit_button("🔄 Reset")
        export_btn = col_export.form_submit_button("📤 Export")

# ===============================
# RESET SESSION
# ===============================
# Jika tombol reset ditekan, maka semua nilai input dikosongkan
if reset_btn:
    for key in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                 "export_data", "last_probas", "last_prediction"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()  # Reload halaman

# ===============================
# PREDIKSI
# ===============================
# Jika tombol submit ditekan, proses prediksi dilakukan
if submit_btn:
    # Ambil input pengguna dan ubah ke bentuk numerik sesuai model
    user_input = {
        "age": st.session_state.age,
        "sex": sex_map[st.session_state.sex],
        "cp": cp_map[st.session_state.cp],
        "trestbps": st.session_state.trestbps,
        "chol": st.session_state.chol,
        "fbs": fbs_map[st.session_state.fbs],
        "restecg": restecg_map[st.session_state.restecg],
        "thalach": st.session_state.thalach,
        "exang": exang_map[st.session_state.exang],
        "oldpeak": st.session_state.oldpeak,
        "slope": slope_map[st.session_state.slope],
        "ca": st.session_state.ca,
        "thal": thal_map[st.session_state.thal]
    }

    df_input = pd.DataFrame([user_input])[features]  # Buat DataFrame input
    prediction = model.predict(df_input)[0]          # Prediksi kelas
    probas = model.predict_proba(df_input)[0]        # Probabilitas dari masing-masing kelas
    label = target_map[prediction]                   # Ubah label hasil ke bentuk teks

    # Simpan hasil prediksi ke session
    st.session_state["export_data"] = {
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "prediction": label,
        "confidence": float(probas[prediction])
    }
    st.session_state["last_probas"] = probas
    st.session_state["last_prediction"] = prediction

# ===============================
# HASIL PREDIKSI
# ===============================
# Menampilkan hasil prediksi jika sudah tersedia
if "last_prediction" in st.session_state and "last_probas" in st.session_state:
    with col_result:
        st.subheader("🎯 Hasil Prediksi")
        pred_label = target_map[st.session_state["last_prediction"]]  # Label hasil
        confidence = st.session_state["last_probas"][st.session_state["last_prediction"]] * 100  # Persentase keyakinan
        st.success(f"Hasil: **{pred_label}** (Keyakinan: {confidence:.2f}%)")

        # Visualisasi Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Keyakinan (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'green' if st.session_state["last_prediction"] == 1 else 'orange'}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=35, b=25))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Visualisasi Bar Chart Probabilitas Kelas
        prob_df = pd.DataFrame({
            "Kelas": list(target_map.values()),
            "Probabilitas": st.session_state["last_probas"]
        })
        fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas", color="Probabilitas",
                         color_continuous_scale=["orange", "green"], title="Distribusi Probabilitas")
        fig_bar.update_layout(height=250, margin=dict(t=25, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

# ===============================
# EXPORT
# ===============================
# Jika tombol export ditekan, hasil prediksi dapat diunduh dalam file JSON
if export_btn:
    if "export_data" in st.session_state:
        st.download_button(
            label="📥 Download Hasil Prediksi",
            data=json.dumps(st.session_state["export_data"], indent=2),
            file_name=f"heart_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ Belum ada hasil prediksi untuk diekspor.")

# ===============================
# FITUR PENTING
# ===============================
# Deskripsi singkat fitur-fitur penting bagi model prediksi penyakit jantung
# cp, ca, thal, thalach, exang, oldpeak, age, slope, trestbps, chol, sex, restecg, fbs

# ===============================
# VISUALISASI FEATURE IMPORTANCE
# ===============================
# Menampilkan pentingnya fitur-fitur dalam model menggunakan grafik horizontal
st.subheader("📊 Pentingnya Fitur dalam Model")
feature_importance = pd.DataFrame({
    "Fitur": features,
    "Pentingnya": model.feature_importances_
}).sort_values("Pentingnya", ascending=True)

fig_feat = px.bar(feature_importance, x="Pentingnya", y="Fitur", orientation="h",
                  color="Pentingnya", color_continuous_scale="viridis",
                  title="Pengaruh Fitur terhadap Prediksi")
fig_feat.update_layout(height=420, margin=dict(t=20, b=40))
st.plotly_chart(fig_feat, use_container_width=True)

# ===============================
# FOOTER
# ===============================
# Informasi pembuat dan afiliasi
st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: gray;'>
    Dibuat menggunakan model <strong>Random Forest</strong> oleh Gabriel Marvel Juan Purwanto<br>
    <em>Fakultas Ilmu Komputer, Universitas Dian Nuswantoro</em> – Streamlit App © 2025
</div>
""", unsafe_allow_html=True)
