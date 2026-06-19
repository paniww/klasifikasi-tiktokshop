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

st.markdown("""
<style>

/* Background utama */
.stApp{
    background-color:#F8FAFC;
}

/* Card Metric */
div[data-testid="metric-container"]{
    background: linear-gradient(
        135deg,
        #60A5FA,
        #38BDF8
    );
    border-radius:15px;
    padding:20px;
    color:white;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

/* Judul metric */
div[data-testid="metric-container"] label{
    color:white !important;
}

/* Nilai metric */
div[data-testid="metric-container"] div{
    color:white !important;
}

/* Tombol */
.stButton > button{
    background: linear-gradient(
        135deg,
        #3B82F6,
        #0EA5E9
    );
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

/* Hover tombol */
.stButton > button:hover{
    background:#2563EB;
}

/* Text Area */
textarea{
    border-radius:10px !important;
    border:2px solid #BFDBFE !important;
}

/* Expander */
.streamlit-expanderHeader{
    background-color:#EFF6FF;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
background:linear-gradient(
135deg,
#60A5FA,
#38BDF8
);
padding:30px;
border-radius:20px;
color:white;
'>

<h1>Analisis Pola Promosi Produk TikTok Shop</h1>

<p>
Implementasi Hasil Penelitian Klasifikasi Akun Buzzer
Menggunakan Algoritma Naïve Bayes
</p>

</div>
""", unsafe_allow_html=True)


st.caption(
    "Implementasi Hasil Penelitian Klasifikasi Akun Buzzer "
    "Menggunakan Algoritma Naïve Bayes"
)

st.write("""
Prototype ini merupakan implementasi hasil penelitian
**Klasifikasi Akun Buzzer di TikTok Shop Menggunakan Algoritma Naïve Bayes
untuk Menganalisis Pola Promosi Produk**.
""")

st.markdown("""
<div style='
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 2px 10px rgba(0,0,0,0.08);
margin-top:20px;
'>

<h3>Tujuan Penelitian</h3>

<p>
Mengklasifikasikan caption TikTok Shop ke dalam kategori
buzzer dan non-buzzer menggunakan algoritma Naïve Bayes
untuk menganalisis pola promosi produk.
</p>

</div>
""", unsafe_allow_html=True)
st.divider()
st.markdown("## Ringkasan Hasil Penelitian")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Dataset", "1.118")
col2.metric("Buzzer", "160")
col3.metric("Non-Buzzer", "958")
col4.metric("Akurasi", "77%")
col5.metric("F1-Score", "81%")

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

st.divider()

st.subheader("Temuan Penelitian")

st.write("""
1. Dataset penelitian terdiri dari 1.118 caption TikTok Shop yang
diklasifikasikan menjadi 160 data buzzer dan 958 data non-buzzer.

2. Caption buzzer cenderung menggunakan bahasa yang lebih persuasif
dan berorientasi pada promosi produk dibandingkan caption non-buzzer.

3. Caption non-buzzer lebih banyak berisi pengalaman penggunaan,
ulasan, dan penyampaian informasi produk.

4. Model Naïve Bayes menghasilkan akurasi sebesar 77%
dalam mengklasifikasikan caption buzzer dan non-buzzer.
""")

st.write("---")

st.info("""
Hasil penelitian menunjukkan bahwa caption buzzer cenderung menggunakan
bahasa persuasif, ajakan pembelian, serta penekanan terhadap promosi produk.
Sebaliknya, caption non-buzzer lebih banyak berisi pengalaman penggunaan,
ulasan, dan informasi produk tanpa ajakan pembelian secara langsung.
""")

st.caption(
    "Prototype penelitian skripsi Program Studi Sistem Informasi "
    "Institut Teknologi Mojosari"
)

st.subheader("Informasi Model")

st.divider()

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

st.divider()

st.markdown("## Analisis Caption Baru")

st.write("""
Masukkan caption TikTok Shop untuk mengetahui
hasil klasifikasi buzzer atau non-buzzer berdasarkan
model hasil penelitian.
""")

caption = st.text_area(
    "Masukkan Caption TikTok Shop",
    height=120
)

if st.button(
    "🔍 Analisis Caption",
    use_container_width=True
):
    if not caption.strip():
        st.warning("Masukkan caption terlebih dahulu.")
        st.stop()

    case_text = case_folding(caption)

    clean_text = cleaning(case_text)

    normal_text = normalisasi(clean_text)

    token_text = tokenizing(normal_text)

    stopword_text = remove_stopwords(token_text)

    stem_text = stemming(stopword_text)

    data = tfidf.transform([stem_text])

    hasil = model.predict(data)[0]
    
    prob = model.predict_proba(data)[0]

    st.divider()
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

    st.divider()
    st.subheader("Probabilitas Klasifikasi")

    st.write(f"Label Non-Buzzer : {prob[0]*100:.2f}%")
    st.write(f"Label Buzzer : {prob[1]*100:.2f}%")

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

