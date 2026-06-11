
import streamlit as st
import joblib

model = joblib.load("model_naive_bayes.pkl")
tfidf = joblib.load("model_tfidf.pkl")

st.title("Klasifikasi Caption TikTok Shop")

caption = st.text_area("Masukkan Caption", height=200)

if st.button("Analisis"):

    data = tfidf.transform([caption])

    hasil = model.predict(data)[0]

    if hasil == 1:
        st.success("BUZZER")
    else:
        st.success("NON-BUZZER")
