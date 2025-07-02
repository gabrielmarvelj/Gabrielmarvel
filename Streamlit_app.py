#import Library 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# ===============================
# PAGE SETUP
# ===============================

#Konfigurasi Halaman Streamlit

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("💓 Prediksi Penyakit Jantung Inspired By - Dr. Eng. Farrikh Alzami, M.Kom")
st.markdown("Aplikasi prediksi risiko penyakit jantung berdasarkan data medis pasien menggunakan model **Random Forest**.")

# ===============================
# LOAD MODEL
# ===============================

# Load Model Machine Learning
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.joblib")

model_bundle = load_model()
model = model_bundle["model"]
features = model_bundle["feature_names"]
target_map = model_bundle["target_map"]

# ===============================
# INPUT MAPPING OPTIONS
# ===============================
#Pemetaan Nilai Input
sex_map = {"Laki-laki": 1, "Perempuan": 0}
cp_map = {"Nyeri saat aktivitas berat": 1, "Nyeri tidak khas": 2, "Nyeri bukan karena jantung": 3, "Tidak ada gejala": 4}
fbs_map = {"Ya (>120 mg/dl)": 1, "Tidak (<=120 mg/dl)": 0}
restecg_map = {"Normal": 0, "ST-T Abnormalitas": 1, "Hypertrophy(pembesaran jantung)": 2}
exang_map = {"Ya": 1, "Tidak": 0}
slope_map = {"Upsloping(meningkat, normal)": 1, "Flat(datar, bisa berisiko)": 2, "Downsloping(menurun, berisiko tinggi)": 3}
thal_map = {"Normal": 3, "Fixed Defect(kerusakan permanen)": 6, "Reversible Defect(kerusakan yang bisa pulih)": 7}

# ===============================
# INPUT FORM
# ===============================
#Form Input Data Pasien

col_form, col_result = st.columns([2, 1])

with col_form:
    st.subheader("📝 Input Data Pasien")
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Umur", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Jenis Kelamin", list(sex_map.keys()))
            cp = st.selectbox("Tipe Nyeri Dada", list(cp_map.keys()))
            trestbps = st.number_input("Tekanan Darah Saat Istirahat (mm Hg)", value=120)
            chol = st.number_input("Kolesterol (mg/dl)", value=240)
            fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl?", list(fbs_map.keys()))
            restecg = st.selectbox("Hasil Elektrokardiografi Saat Istirahat", list(restecg_map.keys()))
        with col2:
            thalach = st.number_input("Detak Jantung Maksimum", value=150)
            exang = st.selectbox("Angina(Nyeridada) akibat Olahraga?", list(exang_map.keys()))
            oldpeak = st.number_input("Oldpeak (Depresi ST)", value=1.0)
            slope = st.selectbox("Kemiringan ST", list(slope_map.keys()))
            ca = st.number_input("Jumlah Pembuluh Darah Utama (0-3)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Tipe Thalassemia", list(thal_map.keys()))
#Tombol Submit, Reset, Export
        col_submit, col_reset, col_export = st.columns([1, 1, 1])
        submit_btn = col_submit.form_submit_button("🔍 Prediksi")
        reset_btn = col_reset.form_submit_button("🔄 Reset")
        export_btn = col_export.form_submit_button("📤 Export")

# ===============================
# HANDLE RESET
# =============================== fungsi reset
if reset_btn:
    st.session_state.clear()
    st.rerun()

# ===============================
# PREDIKSI
# =============================== proses prediksi
if submit_btn:
    user_input = {
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": thalach,
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal_map[thal]
    }

    df_input = pd.DataFrame([user_input])[features]
    prediction = model.predict(df_input)[0]
    probas = model.predict_proba(df_input)[0]

    label = target_map[prediction]
#Tampilkan Hasil Prediksi
    with col_result:
        st.subheader("🎯 Hasil Prediksi")
        st.success(f"Hasil: **{label}** (Keyakinan: {probas[prediction]*100:.2f}%)")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probas[prediction] * 100,
            title={'text': "Keyakinan (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'green' if prediction == 1 else 'orange'}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=35, b=25))
        st.plotly_chart(fig_gauge, use_container_width=True)

        prob_df = pd.DataFrame({"Kelas": list(target_map.values()), "Probabilitas": probas})
        fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas", color="Probabilitas",
                         color_continuous_scale=["orange", "green"], title="Distribusi Probabilitas")
        fig_bar.update_layout(height=250, margin=dict(t=25, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Store prediction in session state
        st.session_state["export_data"] = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "prediction": label,
            "confidence": float(probas[prediction])
        }

# ===============================
# EXPORT
# =============================== Export Hasil Prediksi
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

#cp      = Chest Pain (Tipe Nyeri Dada)
#ca      = Number of Major Vessels Colored by Fluoroscopy (Jumlah pembuluh darah utama yang terlihat dengan fluoroskopi)
#thal    = Thalassemia Type (Tipe thalassemia)
#thalach = Maximum Heart Rate Achieved (Detak jantung maksimum)
#exang   = Exercise Induced Angina (Nyeri dada akibat olahraga)
#oldpeak = ST Depression Induced by Exercise (Penurunan segmen ST akibat olahraga)
#age     = Age (Umur)
#slope   = Slope of the ST Segment (Kemiringan segmen ST saat olahraga)
#trestbps= Resting Blood Pressure (Tekanan darah saat istirahat)
#chol    = Serum Cholesterol (Kadar kolesterol)
#sex     = Sex (Jenis kelamin)
#restecg = Resting Electrocardiographic Results (Hasil EKG saat istirahat)
#fbs     = Fasting Blood Sugar > 120 mg/dl (Gula darah puasa lebih dari 120 mg/dl)

#Visualisasi Feature Importance

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
st.markdown("""
---
<div style='text-align: center; font-size: 14px; color: gray;'>
    Dibuat menggunakan model <strong>Random Forest</strong> oleh Gabriel Marvel Juan Purwanto<br>
    <em>Fakultas Ilmu Komputer, Universitas Dian Nuswantoro</em> – Streamlit App © 2025
</div>
""", unsafe_allow_html=True)
