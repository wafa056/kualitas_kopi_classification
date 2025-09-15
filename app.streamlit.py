import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_kualitas_kopi.joblib")

st.title("Klasifikasi Kualitas Kopi")
st.markdown("Prediksi Kualitas Kopi berdasarkan Kadar Kafein, Tingkat Keasaman dan Jenis Proses")

kadar_kafein = st.slider("Kadar Kafein", 50.0, 200.0, 110.0)
tingkat_keasaman = st.slider("tingkat Keasaman", 1.0, 7.0, 5.0)
jenis_proses = st.pills("Jenis Proses",
		["Natural", "Honey", "Washed"], default="Natural")

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[kadar_kafein, tingkat_keasaman, jenis_proses]], columns=["Kadar Kafein", "Tingkat Keasaman", "Jenis Proses",])

	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Prediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :coffee: oleh Wafa' Nailatur Rokhmah")
