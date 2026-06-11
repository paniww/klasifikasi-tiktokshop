import streamlit as st
import joblib

model = joblib.load("model_naive_bayes.pkl")
tfidf = joblib.load("model_tfidf.pkl")

st.title("Sistem Klasifikasi Indikasi Akun Buzzer pada TikTok Shop")

st.write(
    "Sistem ini mengidentifikasi indikasi akun buzzer berdasarkan caption promosi produk yang dipublikasikan pada TikTok Shop menggunakan metode TF-IDF dan algoritma Naïve Bayes."
)

caption = st.text_area(
    "Masukkan Caption",
    height=200
)

if st.button("Analisis"):

    data = tfidf.transform([caption])

    hasil = model.predict(data)[0]

    prob = model.predict_proba(data)[0]

    st.subheader("Hasil Klasifikasi")

    if hasil == 1:
        st.error("BUZZER")
    else:
        st.success("NON-BUZZER")

    st.write("---")

    st.write(f"Probabilitas Label 0 : {prob[0]*100:.2f}%")
    st.write(f"Probabilitas Label 1 : {prob[1]*100:.2f}%")
