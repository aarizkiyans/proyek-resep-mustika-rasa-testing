import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

# Import LangChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Chef AI Mustikarasa",
    page_icon="🥘",
    layout="centered"
)

st.title("🥘 Detektif Resep Mustikarasa")
st.markdown("---")

# ==========================================
# 2. SETUP API KEY (VERSI ANTI-GAGAL)
# ==========================================
# Logika: Cek dulu di secrets (file). Kalau gak ada, minta input manual di layar.
api_key = None

try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    pass # Lanjut ke bawah kalau file secrets gak ada

# Kalau API Key belum ketemu di secrets, tampilkan input di Sidebar
if not api_key:
    st.warning("⚠️ API Key tidak ditemukan di secrets.toml")
    api_key = st.sidebar.text_input("Masukkan Google API Key Anda:", type="password")

# Kalau user belum masukin key, stop dulu programnya biar gak error
if not api_key:
    st.info("👈 Silakan masukkan Google API Key di menu sebelah kiri untuk memulai.")
    st.stop()

# Set Environment Variable
os.environ["GOOGLE_API_KEY"] = api_key

# ==========================================
# 3. SETUP MODEL & DATABASE (CACHE)
# ==========================================
# Cache resource biar gak loading ulang tiap kali klik tombol

@st.cache_resource
# --- 3. LOAD MODEL GAMBAR (DARI GOOGLE DRIVE) ---
@st.cache_resource
def load_image_model():
    # Nama file lokal (nanti disave dengan nama ini di server)
    local_model_path = "/content/drive/MyDrive/Dataset Indonesian Food /Indonesian_Food_VGG16.keras"

    # Cek: Kalau file belum ada di server, download dulu
    if not os.path.exists(local_model_path):
        # MASUKKAN FILE ID DARI GOOGLE DRIVE KAMU DI SINI!
        # Contoh: file_id = "1A-BcDeF...xyz"
        file_id = "1Odf--QVAO-q2REy0EdKiqxMU3ufRps4Y" 
        
        url = f'https://drive.google.com/uc?id={file_id}'
        
        print("Sedang mendownload model dari Google Drive...")
        gdown.download(url, local_model_path, quiet=False)
        print("✅ Download selesai!")

    # Load model setelah file tersedia
    return tf.keras.models.load_model(local_model_path)

@st.cache_resource
def load_rag_system():
    # 1. Setup Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 2. Load Database (Pastikan folder 'chroma_db' ada)
    # allow_dangerous_deserialization=True diperlukan untuk versi Chroma terbaru
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # 3. Setup Otak (Gemini Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 4. Setup Chain (Rantai Proses)
    retriever = vectorstore.as_retriever()
    
    template = """Kamu adalah asisten ahli masakan Indonesia dari buku legendaris Mustikarasa.
Gunakan potongan konteks resep di bawah ini untuk menjawab pertanyaan pengguna.
Jika pertanyaannya bukan soal resep, berikan output yang sesuai dengan isi buku mustika rasa. 
Kalau pertanyaanya ada nama daerahnya misal rendang padang tapi gak ada spesifik rendang padang, cari aja resep makananan yang berhubungan dengan rendang atau padangnya

KONTEKS RESEP:
{context}

PERTANYAAN USER:
{question}

INSTRUKSI:
1. Jawablah dengan ramah dan lengkap.
2. Jika ada bahan dan takaran, sebutkan dengan jelas.
3. Jangan mengarang resep sendiri! Gunakan HANYA informasi dari konteks di atas.
4. Jika resep tidak ditemukan di konteks, katakan jujur bahwa tidak ada di buku ini.

JAWABAN:
    """
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Load sistem saat aplikasi dibuka pertama kali
with st.spinner("Sedang menyiapkan dapur AI..."):
    try:
        classifier_model = load_image_model()
        rag_chain = load_rag_system()
        st.success("✅ Sistem Siap! Silakan upload foto.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat loading sistem: {e}")
        st.stop()

# ==========================================
# 4. INTERFACE UTAMA
# ==========================================

uploaded_file = st.file_uploader("Upload foto makanan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # A. Tampilkan Gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Foto Makanan', use_container_width=True)
    
    # B. Proses Prediksi Gambar
    st.write("🔍 Sedang menerawang jenis makanan...")
    
    # Preprocessing (Sesuai VGG16 Training kamu)
    target_size = (224, 224)
    image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    img_array = img_array / 255.0 # Normalisasi
    img_batch = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
    
    # Prediksi
    predictions = classifier_model.predict(img_batch)
    idx_max = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # --- PENTING: LIST KELAS MAKANAN ---
    # GANTI LIST INI SESUAI URUTAN FOLDER DATASET KAMU!
    # Contoh: Kalau folder trainingnya [Ayam, Rendang, Sate], urutannya harus sama.
    CLASS_NAMES = ["bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng", "pempek", 
               "rawon", "rendang", "sate", "soto"] 
    
    predicted_label = CLASS_NAMES[idx_max]
    
    # Tampilkan Hasil Deteksi
    st.info(f"🤖 Saya yakin **{confidence:.1f}%** ini adalah: **{predicted_label}**")
    
    # C. Tombol Cari Resep (RAG)
    if st.button(f"📖 Cari Resep {predicted_label} Sekarang!"):
        with st.spinner(f"Membuka buku Mustikarasa halaman {predicted_label}..."):
            try:
                # Query ke RAG
                query = f"Berikan resep lengkap cara membuat {predicted_label} beserta bahan-bahannya sesuai buku."
                hasil_resep = rag_chain.invoke(query)
                
                # Tampilkan Resep
                st.markdown("---")
                st.subheader(f"📜 Resep Asli: {predicted_label}")
                st.markdown(hasil_resep)
                
            except Exception as e:
                st.error(f"Gagal mencari resep: {e}")