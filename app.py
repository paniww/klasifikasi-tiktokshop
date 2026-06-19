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

.stApp{
    background-color:#F8FAFC;
}

section[data-testid="stSidebar"]{
    background-color:#E0F2FE;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    border-left:5px solid #38BDF8;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.stButton > button{
    background:#38BDF8;
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#0EA5E9;
}

</style>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Dataset",
        "Preprocessing",
        "TF-IDF",
        "Evaluasi Model",
        "Pola Promosi",
        "Analisis Caption"
    ]
)

# ==================================
# DASHBOARD
# ==================================

if menu == "Dashboard":

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#38BDF8,#60A5FA);
    padding:30px;
    border-radius:20px;
    color:white;
    ">
    <h1>Analisis Pola Promosi Produk TikTok Shop</h1>
    <p>
    Implementasi Klasifikasi Akun Buzzer Menggunakan
    Algoritma Naïve Bayes
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.title("Analisis Pola Promosi Produk TikTok Shop")

    st.write("""
    Sistem ini merupakan implementasi hasil penelitian
    klasifikasi akun buzzer pada TikTok Shop menggunakan
    algoritma Naïve Bayes untuk menganalisis pola promosi produk.
    """)

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric("Dataset","1.118")
    col2.metric("Buzzer","160")
    col3.metric("Non-Buzzer","958")
    col4.metric("Akurasi","77%")
    col5.metric("F1-Score","81%")

    st.subheader("Ringkasan Hasil Penelitian")

    st.write("""
    Penelitian dilakukan terhadap 1.118 caption TikTok Shop.
    Hasil klasifikasi menunjukkan bahwa caption buzzer
    cenderung menggunakan bahasa persuasif dan berorientasi
    pada promosi produk, sedangkan caption non-buzzer lebih
    banyak berisi pengalaman penggunaan dan ulasan produk.
    """)

    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📊 Lihat Hasil Penelitian",
            use_container_width=True
        ):
            st.info("""
            Silakan pilih menu Dataset,
            Preprocessing, TF-IDF,
            Evaluasi Model, atau
            Pola Promosi pada sidebar.
            """)
    with col2:
        if st.button(
            "🔍 Mulai Analisis Caption",
            use_container_width=True
        ):
            st.info("""
            Silakan pilih menu
            Analisis Caption
            pada sidebar.
            """)

# ==================================
# DATASET
# ==================================

elif menu == "Dataset":

    st.header("Dataset Penelitian")

    st.write("""
    Dataset penelitian terdiri dari 1.118 caption TikTok Shop
    yang telah melalui proses seleksi dan pelabelan data.
    """)

    col1,col2 = st.columns(2)

    col1.metric("Buzzer","160")
    col2.metric("Non-Buzzer","958")

    st.subheader("Distribusi Dataset")

    st.write("""
    Label 1 : Buzzer (160 data)

    Label 0 : Non-Buzzer (958 data)
    """)
    st.divider()
    st.markdown(""" 
    ### Deskripsi Dataset

    Dataset penelitian terdiri dari 1.118 caption
    TikTok Shop yang diperoleh melalui proses
    pengumpulan data, seleksi data,
    preprocessing, dan pelabelan manual.

    Data kemudian dibagi menjadi
    894 data latih dan 224 data uji
    untuk proses klasifikasi menggunakan
    algoritma Naïve Bayes.
    """)




# ==================================
# PREPROCESSING
# ==================================

elif menu == "Preprocessing":

    st.header("Tahapan Preprocessing")

    st.write("""
    Tahapan preprocessing yang digunakan dalam penelitian:
    """)

    st.markdown("""
    1. Case Folding
    2. Cleaning
    3. Normalisasi
    4. Tokenizing
    5. Stopword Removal
    6. Stemming
    """)

    st.image("alur_preprocessing.png")

    st.subheader("Tahapan yang Digunakan")
    tahapan_df = pd.DataFrame({
        "Tahap":[
            "Case Folding",
            "Cleaning",
            "Normalisasi",
            "Tokenizing",
            "Stopword Removal",
            "Stemming"
        ]
    })

    st.table(tahapan_df)

    st.info("""
    Tahapan preprocessing bertujuan untuk membersihkan
    dan menyeragamkan teks sehingga dapat digunakan
    pada proses ekstraksi fitur TF-IDF.
    """)

# ==================================
# TF-IDF
# ==================================

elif menu == "TF-IDF":

    st.header("Ekstraksi Fitur TF-IDF")

    st.write("""
    Ekstraksi fitur dilakukan menggunakan metode TF-IDF
    untuk mengubah data teks menjadi representasi numerik.
    """)

    contoh = pd.DataFrame({
        "Kata":[
            "banget",
            "produk",
            "promo",
            "murah",
            "review"
        ],
        "Bobot TF-IDF":[
            0.421,
            0.387,
            0.355,
            0.311,
            0.287
        
        ]
    })

    st.table(contoh)

# ==================================
# EVALUASI
# ==================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy","77%")
    col2.metric("Precision","76%")
    col3.metric("Recall","85%")
    col4.metric("F1 Score","81%")

    st.image("confusion_matrix.png")

    st.success("""
    Model Naïve Bayes menghasilkan:
    
    Accuracy : 77%
    Precision : 76%
    Recall : 85%
    F1-Score : 81%
    """)

    st.write("""
    Hasil pengujian menunjukkan bahwa model Naïve Bayes
    mampu mengklasifikasikan caption buzzer dan non-buzzer
    dengan performa yang cukup baik.
    """)

# ==================================
# POLA PROMOSI
# ==================================

elif menu == "Pola Promosi":

    st.header("Analisis Pola Promosi Produk")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Karakteristik Buzzer")

        st.write("""
        - promo
        - beli
        - murah
        - viral
        - diskon
        - keranjang
        - link
        - harga
        """)

    with col2:

        st.subheader("Karakteristik Non-Buzzer")

        st.write("""
        - review
        - produk
        - pakai
        - skincare
        - kulit
        - barang
        - coba
        """)

    st.subheader("Temuan Penelitian")

    st.write("""
    Caption buzzer cenderung menggunakan bahasa persuasif,
    ajakan pembelian, serta penekanan terhadap manfaat produk.

    Sebaliknya, caption non-buzzer lebih banyak berisi
    pengalaman penggunaan dan ulasan produk.
    """)

    st.success("""
    Pola promosi akun buzzer ditandai dengan:

    • Penggunaan kata promosi yang kuat
    
    • Ajakan pembelian secara langsung

    • Penekanan manfaat produk

    • Bahasa yang lebih persuasif

    Sedangkan akun non-buzzer
    lebih berfokus pada ulasan
    dan pengalaman penggunaan produk.
    """)


# ==================================
# ANALISIS CAPTION
# ==================================

elif menu == "Analisis Caption":

    st.header("Analisis Caption Baru")

    st.write("""
    Masukkan caption TikTok Shop
    untuk mengetahui hasil klasifikasi
    buzzer atau non-buzzer berdasarkan
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

        # HASIL KLASIFIKASI
        st.subheader("Hasil Klasifikasi")

        if hasil == 1:
            st.error("⚠️ BUZZER")
        else:
            st.success("✅ NON-BUZZER")

        # PROBABILITAS
        st.subheader("Probabilitas Klasifikasi")

        st.write(f"Label Non-Buzzer : {prob[0]*100:.2f}%")
        st.write(f"Label Buzzer : {prob[1]*100:.2f}%")

        # PREPROCESSING
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

        # TF-IDF
        st.subheader("Hasil Ekstraksi Fitur TF-IDF")

        feature_names = tfidf.get_feature_names_out()
        scores = data.toarray()[0]

        hasil_tfidf = []

        for kata, skor in zip(feature_names, scores):
            if skor > 0:
                hasil_tfidf.append(
                    (kata, round(float(skor), 4))
                )

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
