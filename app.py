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

    st.subheader("Deskripsi Sistem")

    st.write("""
    Sistem ini merupakan implementasi hasil penelitian
    klasifikasi akun buzzer pada TikTok Shop menggunakan
    algoritma Naïve Bayes untuk menganalisis pola promosi produk.
    """)

    col1,col2,col3,col4,col5 = st.columns(5)

    col1.metric("Dataset","1.118")
    col2.metric("Buzzer","637")
    col3.metric("Non-Buzzer","481")
    col4.metric("Akurasi","77%")
    col5.metric("F1-Score","81%")

    st.subheader("Ringkasan Hasil Penelitian")

    st.write("""
    Penelitian ini menggunakan 1.118 caption TikTok Shop yang
    telah dilabeli menjadi 637 data buzzer dan 481 data non-buzzer.
    Hasil pengujian menunjukkan bahwa model Naïve Bayes memperoleh
    akurasi sebesar 77% dan F1-score sebesar 81%.
    Berdasarkan hasil klasifikasi, caption buzzer cenderung
    menggunakan bahasa persuasif dan ajakan pembelian,
    sedangkan caption non-buzzer lebih banyak berisi ulasan
    dan pengalaman penggunaan produk.
    """)
    st.markdown("""
    1. Mengklasifikasikan caption buzzer dan non-buzzer pada TikTok Shop menggunakan algoritma Naïve Bayes.
    2. Menganalisis pola promosi produk berdasarkan hasil klasifikasi caption.
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

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Dataset", "1.118")
    col2.metric("Buzzer", "637")
    col3.metric("Non-Buzzer", "481")

    st.subheader("Distribusi Dataset")

    st.table(pd.DataFrame({
        "Label":[1,0],
        "Kategori":["Buzzer","Non-Buzzer"],
        "Jumlah":[637,481]
    }))
    
    col1, col2 = st.columns(2)

    col1.metric("Persentase Buzzer", "56,98%")
    col2.metric("Persentase Non-Buzzer", "43,02%")
    st.subheader("Pembagian Data")

    col1, col2 = st.columns(2)

    col1.metric("Data Latih (80%)", "894")
    col2.metric("Data Uji (20%)", "224")
    
    st.divider()
    st.markdown(""" 
    ### Deskripsi Dataset

    Dataset penelitian terdiri dari 1.118 caption TikTok Shop yang diperoleh melalui proses pengumpulan data, seleksi data, dan pelabelan manual. Dataset tersebut terdiri dari 637 data buzzer dan 481 data non-buzzer yang digunakan dalam proses pelatihan dan pengujian model.
    """)

# ==================================
# PREPROCESSING
# ==================================

elif menu == "Preprocessing":

    st.header("Tahapan Preprocessing")

    st.write("""
    Tahapan preprocessing yang digunakan dalam penelitian:
    """)

    st.subheader("Tahapan yang Digunakan")
    tahapan_df = pd.DataFrame({
        "Tahap":[
            "Case Folding",
            "Cleaning",
            "Normalisasi",
            "Tokenizing",
            "Stopword Removal",
            "Stemming"
        ],
        "Keterangan":[
            "Mengubah seluruh huruf menjadi huruf kecil.",
            "Menghapus URL, hashtag, mention, angka, dan karakter yang tidak diperlukan.",
            "Mengubah kata tidak baku menjadi kata baku sesuai kamus normalisasi.",
            "Memecah kalimat menjadi token atau kata-kata tunggal.",
            "Menghapus kata yang tidak memiliki pengaruh signifikan terhadap klasifikasi.",
            "Mengubah kata berimbuhan menjadi bentuk kata dasar."
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
    Ekstraksi fitur TF-IDF digunakan untuk mengubah data caption
    hasil preprocessing menjadi representasi numerik yang dapat
    digunakan oleh algoritma Naïve Bayes dalam proses klasifikasi.
    """)
    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", "1.118")
    col2.metric("Metode", "TF-IDF")
    col3.metric("Fitur Teks", "Representasi Numerik")
    
    st.subheader("Contoh Hasil Ekstraksi TF-IDF")

    contoh = pd.DataFrame({
        "Username":[
            "fluffyaa",
            "sffhjiri",
            "cempaka465",
            "dokterdetektifhero",
            "review_skincare"
        ],
        "Pakai":[
            0,
            0,
            0.1304,
            0.0404,
            0.2552
        ],
        "Hasil":[
            0,
            0.1102,
            0,
            0.0764,
            0
        ],
        "Cakep":[
            0.4809,
            0,
            0,
            0,
            0.2859
        ]
    })
    
    st.dataframe(
        contoh,
        use_container_width=True
    )
    
    st.info("""
    Tabel di atas menunjukkan hasil ekstraksi fitur TF-IDF
    dari beberapa caption TikTok Shop. Nilai TF-IDF yang
    lebih besar menunjukkan bahwa suatu kata memiliki
    kontribusi yang lebih tinggi terhadap representasi
    dokumen dibandingkan kata lainnya.
    """)

# ==================================
# EVALUASI
# ==================================

elif menu == "Evaluasi Model":

    st.header("Evaluasi Model")
    
    st.write("""
    Evaluasi model dilakukan menggunakan confusion matrix
    serta metrik accuracy, precision, recall, dan F1-score
    untuk mengukur performa klasifikasi caption buzzer dan non-buzzer.
    """)
    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "77%")
    col2.metric("Precision", "76%")
    col3.metric("Recall", "85%")
    col4.metric("F1 Score", "81%")

    st.subheader("Confusion Matrix")

    st.image(
        "confusion_matrix.png",
        use_container_width=True
    )

    st.subheader("Ringkasan Confusion Matrix")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("TP", "110")
    col2.metric("TN", "63")
    col3.metric("FP", "33")
    col4.metric("FN", "18")

    st.success("""
    Model Naïve Bayes menghasilkan performa klasifikasi yang cukup baik
    dengan Accuracy 77%, Precision 76%, Recall 85%, dan F1-Score 81%.
    """)

    st.info("""
    Accuracy menunjukkan tingkat ketepatan model secara keseluruhan.
    Precision menunjukkan ketepatan model dalam memprediksi kelas buzzer.
    Recall menunjukkan kemampuan model dalam mengenali data buzzer.
    F1-Score merupakan kombinasi antara precision dan recall.
    """)

    st.write("""
    Berdasarkan hasil evaluasi, model memiliki nilai recall sebesar 85%
    yang menunjukkan bahwa model mampu mengenali sebagian besar caption
    buzzer dengan baik. Nilai accuracy sebesar 77% menunjukkan bahwa
    model dapat mengklasifikasikan caption buzzer dan non-buzzer dengan
    performa yang cukup baik.
    """)

# ==================================
# POLA PROMOSI
# ==================================

elif menu == "Pola Promosi":
elif menu == "Pola Promosi":

    st.header("Analisis Pola Promosi Produk")

    st.write("""
    Halaman ini menampilkan hasil analisis pola promosi produk
    berdasarkan caption yang telah diklasifikasikan menggunakan
    algoritma Naïve Bayes.
    """)

    st.subheader("Karakteristik Caption")

    karakteristik = pd.DataFrame({
        "Buzzer":[
            "promo",
            "beli",
            "murah",
            "viral",
            "diskon",
            "keranjang",
            "link",
            "harga"
        ],
        "Non-Buzzer":[
            "review",
            "produk",
            "pakai",
            "skincare",
            "kulit",
            "barang",
            "coba",
            "-"
        ]
    })

    st.table(karakteristik)

    st.subheader("Hasil Analisis Pola Promosi Produk")

    hasil_pola = pd.DataFrame({
        "Kategori":[
            "Buzzer",
            "Non-Buzzer"
        ],
        "Karakteristik":[
            "Menggunakan kata promosi, ajakan pembelian, dan penekanan manfaat produk",
            "Berisi ulasan, pengalaman penggunaan, dan opini pengguna"
        ]
    })

    st.table(hasil_pola)

    st.subheader("Temuan Penelitian")

    st.info("""
    Caption buzzer cenderung menggunakan bahasa persuasif,
    ajakan pembelian secara langsung, serta penekanan terhadap
    keunggulan produk. Sebaliknya, caption non-buzzer lebih
    banyak berisi pengalaman penggunaan, ulasan produk,
    dan informasi yang bersifat deskriptif.
    """)

    st.success("""
    Berdasarkan hasil klasifikasi, pola promosi akun buzzer
    ditandai dengan penggunaan kata promosi yang kuat,
    ajakan pembelian secara langsung, dan penekanan manfaat produk.
    Sementara itu, akun non-buzzer lebih berfokus pada ulasan
    serta pengalaman penggunaan produk.
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
