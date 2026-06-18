import streamlit as st
import joblib

model = joblib.load("model_naive_bayes.pkl")
tfidf = joblib.load("model_tfidf.pkl")

st.set_page_config(page_title="Analisis Pola Promosi Produk TikTok Shop")

st.title("Analisis Pola Promosi Produk TikTok Shop")
st.subheader("Berdasarkan Klasifikasi Buzzer dan Non-Buzzer")

st.write("""
Prototype ini merupakan implementasi hasil penelitian
klasifikasi buzzer dan non-buzzer pada TikTok Shop
menggunakan metode TF-IDF dan algoritma Naïve Bayes.
""")

caption = st.text_area(
"Masukkan Caption",
height=200
)

if st.button("Analisis"):

```
data = tfidf.transform([caption])

hasil = model.predict(data)[0]

prob = model.predict_proba(data)[0]

st.subheader("Hasil Klasifikasi")

if hasil == 1:

    st.error("BUZZER")

    st.write("""
    Karakteristik:
    - Mengandung unsur promosi yang kuat
    - Menekankan manfaat produk
    - Terdapat ajakan pembelian
    - Bersifat persuasif
    """)

else:

    st.success("NON-BUZZER")

    st.write("""
    Karakteristik:
    - Berisi pengalaman penggunaan produk
    - Bersifat informatif
    - Berupa ulasan produk
    - Tidak ditemukan ajakan pembelian langsung
    """)

st.write("---")

st.subheader("Probabilitas Klasifikasi")

st.write(f"Label Non-Buzzer : {prob[0]*100:.2f}%")
st.write(f"Label Buzzer : {prob[1]*100:.2f}%")
```

st.write("---")

st.subheader("Informasi Model")

st.write("""

* Algoritma : Naïve Bayes
* Ekstraksi Fitur : TF-IDF
* Jumlah Data : 1.118 Data
* Data Latih : 894 Data
* Data Uji : 224 Data
* Akurasi : 77%
  """)

st.write("---")

st.subheader("Hasil Analisis Pola Promosi")

col1, col2 = st.columns(2)

with col1:
st.markdown("### Karakteristik Buzzer")
st.write("""
banget,
beli,
murah,
viral,
promo,
keranjang,
link,
harga
""")

with col2:
st.markdown("### Karakteristik Non-Buzzer")
st.write("""
review,
jujur,
pakai,
skincare,
kulit,
produk,
barang
""")

st.write("---")

st.info("""
Hasil penelitian menunjukkan bahwa caption buzzer
cenderung menggunakan bahasa persuasif dan ajakan
pembelian secara langsung, sedangkan caption non-buzzer
lebih berfokus pada pengalaman penggunaan dan ulasan produk.
""")
