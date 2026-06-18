import streamlit as st
import joblib
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

model = joblib.load("model_naive_bayes.pkl")
tfidf = joblib.load("model_tfidf.pkl")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def case_folding(text):
    return str(text).lower()

def cleaning(text):

    text = str(text)

    # hapus URL
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # hapus mention
    text = re.sub(r'@\w+', ' ', text)

    # hapus hashtag
    text = re.sub(r'#\w+', ' ', text)

    # hapus angka
    text = re.sub(r'\d+', ' ', text)

    # hapus tanda baca & karakter selain huruf
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

normalisasi_dict = {

    'yg':'yang',
    'ga':'tidak',
    'gak':'tidak',
    'nggak':'tidak',
    'g':'tidak',
    'dunk':'dong',

    'udah':'sudah',
    'aja':'saja',
    'bgt':'banget',

    'kalo':'kalau',
    'kl':'kalau',

    'sampe':'sampai',
    'ngasi':'memberi',

    'pake':'pakai',
    'kepake':'terpakai',
    'dipake':'dipakai',

    'tau':'tahu',
    'tp':'tapi',
    'tuh':'itu',

    'gue':'saya',
    'gw':'saya',

    'nih':'ini',
    'ni':'ini',
    'yaa':'ya'
}

def normalisasi(text):

    hasil = []

    for kata in text.split():

        hasil.append(
            normalisasi_dict.get(kata, kata)
        )

    return " ".join(hasil)

def tokenizing(text):
    return text.split()

stop_words = set(
    stopwords.words('indonesian')
)

custom_stopwords = {

    'aku',
    'saya',
    'kalian',
    'kamu',

    'ya',
    'sih',

    'nih',
    'dong',

    'to',
    'it',
    'the',
    'my',

    'nya'
}

stop_words.update(custom_stopwords)

def remove_stopwords(tokens):

    return [
        kata
        for kata in tokens
        if kata not in stop_words
    ]

def stemming(tokens):
    hasil = []
    for kata in tokens:
        hasil.append(stemmer.stem(kata))
    return " ".join(hasil)

st.set_page_config(
    page_title="Analisis Pola Promosi Produk TikTok Shop",
    layout="wide"
)

st.title("Prototype Analisis Pola Promosi Produk TikTok Shop")
st.caption(
    "Implementasi Klasifikasi Buzzer dan Non-Buzzer "
    "Menggunakan Algoritma Naïve Bayes"
)

st.write("""
Prototype ini merupakan implementasi hasil penelitian
**Klasifikasi Akun Buzzer di TikTok Shop Menggunakan Algoritma Naïve Bayes
untuk Menganalisis Pola Promosi Produk**.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", "1.118")

with col2:
    st.metric("Data Latih", "894")

with col3:
    st.metric("Data Uji", "224")

caption = st.text_area(
    "Masukkan Caption TikTok Shop",
    height=120
)

if st.button(
    "🔍 Analisis Caption",
    use_container_width=True
)
    if not caption.strip():
        st.warning("Masukkan caption terlebih dahulu.")
        st.stop()

    case_text = case_folding(caption)

    clean_text = cleaning(case_text)

    normal_text = normalisasi(clean_text)

    token_text = tokenizing(normal_text)

    stopword_text = remove_stopwords(token_text)

    stem_text = stemming(stopword_text)

    st.divider()
    st.subheader("Tahapan Preprocessing")

    with st.expander("0. Caption Asli"):
        st.write(caption)

    with st.expander("1. Case Folding"):
        st.write(case_text)

    with st.expander("2. Cleaning"):
        st.write(clean_text)

    with st.expander("3. Normalisasi"):
        st.write(normal_text)

    with st.expander("4. Tokenizing"):
        st.write(token_text)

    with st.expander("5. Stopword Removal"):
        st.write(stopword_text)

    with st.expander("6. Stemming"):
        st.write(stem_text)

    data = tfidf.transform([stem_text])

    st.subheader("Hasil Ekstraksi Fitur TF-IDF")

    feature_names = tfidf.get_feature_names_out()

    scores = data.toarray()[0]

    hasil_tfidf = []

    for kata, skor in zip(feature_names, scores):
        if skor > 0:
            hasil_tfidf.append((kata, round(float(skor), 4)))

    hasil_tfidf = sorted(
        hasil_tfidf,
        key=lambda x: x[1],
        reverse=True
    )

    if len(hasil_tfidf) > 0:

        df_tfidf = pd.DataFrame(
           hasil_tfidf[:10],
           columns=["Kata", "Bobot TF-IDF"]
    )

        st.table(df_tfidf)

    else:
        st.write("Tidak ada fitur TF-IDF yang terdeteksi.")

    hasil = model.predict(data)[0]

    prob = model.predict_proba(data)[0]

    st.subheader("Hasil Klasifikasi")

    if hasil == 1:
        st.error("⚠️ BUZZER")

        st.write("""
        Berdasarkan hasil klasifikasi model, caption ini memiliki
        karakteristik promosi yang umum ditemukan pada kategori buzzer.
        """)

    else:

        st.success("✅ NON-BUZZER")

        st.write("""
        Berdasarkan hasil klasifikasi model, caption ini lebih
        menunjukkan pola ulasan dan pengalaman penggunaan produk.
        """)

    st.subheader("Probabilitas Klasifikasi")

    st.write(f"Label Non-Buzzer : {prob[0]*100:.2f}%")
    st.write(f"Label Buzzer : {prob[1]*100:.2f}%")

st.write("---")

st.subheader("Informasi Model")

st.markdown("""
- **Algoritma** : Naïve Bayes
- **Ekstraksi Fitur** : TF-IDF
- **Jumlah Data** : 1.118 Data
- **Data Latih** : 894 Data
- **Data Uji** : 224 Data
- **Akurasi** : 77%
- **Precision** : 76%
- **Recall** : 85%
- **F1-Score** : 81%
""")

st.write("---")

st.subheader("Hasil Analisis Pola Promosi")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Karakteristik Caption Buzzer")

    st.write("""
    - banget
    - beli
    - murah
    - viral
    - promo
    - keranjang
    - link
    - harga
    """)

with col2:
    st.markdown("### Karakteristik Caption Non-Buzzer")

    st.write("""
    - review
    - jujur
    - pakai
    - skincare
    - kulit
    - produk
    - barang
    """)

st.write("---")

st.info("""
Hasil penelitian menunjukkan bahwa caption buzzer cenderung menggunakan
bahasa persuasif, ajakan pembelian, serta penekanan terhadap promosi produk.
Sebaliknya, caption non-buzzer lebih banyak berisi pengalaman penggunaan,
ulasan, dan informasi produk tanpa ajakan pembelian secara langsung.
""")
st.subheader("Temuan Penelitian")

st.write("""
Kata yang paling sering muncul pada kategori buzzer
antara lain 'banget', 'beli', 'murah', 'viral', dan 'promo'.

Sementara pada kategori non-buzzer lebih banyak ditemukan
kata yang berkaitan dengan pengalaman penggunaan dan ulasan produk.
""")

st.caption(
    "Prototype penelitian skripsi Program Studi Sistem Informasi "
    "Institut Teknologi Mojosari"
)
