import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

# Import LangChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. KONFIGURASI HALAMAN & API KEY
# ==========================================
st.set_page_config(
    page_title="Chef AI Mustikarasa",
    page_icon="🥘",
    layout="centered"
)

st.title("🥘 Detektif Resep Mustikarasa")
st.write("Temukan resep legendaris dari foto atau nama makanannya langsung!")
st.markdown("---")

# --- SETUP API KEY (VERSI ANTI-GAGAL) ---
api_key = None

try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    pass 

if not api_key:
    st.warning("⚠️ API Key tidak ditemukan di secrets.toml")
    api_key = st.sidebar.text_input("Masukkan Google API Key Anda:", type="password")

if not api_key:
    st.info("👈 Silakan masukkan Google API Key di menu sebelah kiri untuk memulai.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key 

# ==========================================
# 2. SETUP MODEL & DATABASE (CACHE)
# ==========================================

# --- FUNGSI BANGUN MODEL MANUAL (SOLUSI ERROR FLATTEN) ---
def build_vgg_architecture():
    # 1. Bangun Wadah VGG16 Kosong
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights=None, 
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False 

    # 2. Susun Layer
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax') # 10 Kelas Makanan
    ])
    return model

@st.cache_resource
def load_image_model():
    local_model_path = "Indonesian_Food_VGG16.keras"

    # --- STEP 1: DOWNLOAD DARI GDRIVE ---
    if not os.path.exists(local_model_path):
        # ID Google Drive Kamu (JANGAN DIGANTI)
        file_id = "1Odf--QVAO-q2REy0EdKiqxMU3ufRps4Y" 
        url = f'https://drive.google.com/uc?id={file_id}'
        
        print(f"Sedang mendownload model ke {local_model_path}...")
        try:
            gdown.download(url, local_model_path, quiet=False)
            print("✅ Download selesai!")
        except Exception as e:
            st.error(f"Gagal download model: {e}")
            st.stop()

    # --- STEP 2: LOAD MANUAL (JURUS SUNTIK) ---
    try:
        model = build_vgg_architecture()
        # Pancing model biar strukturnya kebentuk
        model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
        
        print("Menyuntikkan bobot ke model...")
        model.load_weights(local_model_path)
        return model
        
    except Exception as e:
        st.error(f"Error fatal saat loading model: {e}")
        st.stop()

@st.cache_resource
def load_rag_system():
    # 1. Setup Embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # 2. Load Database
    vectorstore = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # 3. Setup Otak (Pakai 1.5 Flash biar stabil)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    
    # 4. Setup Chain
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

# Load sistem saat aplikasi dibuka
with st.spinner("Sedang menyiapkan dapur AI..."):
    try:
        classifier_model = load_image_model()
        rag_chain = load_rag_system()
        st.success("✅ Sistem Siap! Pilih metode input di bawah.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat loading sistem: {e}")
        st.stop()

# ==========================================
# 3. FUNGSI PENCARI RESEP (LOGIKA RAG)
# ==========================================
def cari_resep_di_buku(nama_makanan):
    st.markdown("---")
    st.subheader(f"📜 Hasil Pencarian: {nama_makanan}")
    
    with st.spinner(f"Membuka buku Mustikarasa mencari resep '{nama_makanan}'..."):
        try:
            query = f"Berikan resep lengkap cara membuat {nama_makanan} beserta bahan-bahannya sesuai buku."
            hasil_resep = rag_chain.invoke(query)
            st.markdown(hasil_resep)
        except Exception as e:
            st.error(f"Gagal mencari resep: {e}")

# ==========================================
# 4. INTERFACE UTAMA (MODE GAMBAR & TEKS)
# ==========================================

# Pilihan Mode
input_mode = st.radio(
    "Pilih cara Anda ingin mencari resep:",
    ("🖼️ Upload Foto Makanan", "✍️ Tulis Nama Makanan"),
    horizontal=True
)
st.markdown("---")

# --- MODE 1: GAMBAR ---
if input_mode == "🖼️ Upload Foto Makanan":
    st.info("Upload foto makanan (JPG/PNG), nanti AI akan menebaknya.")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Tampilkan Gambar
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Foto yang diupload', width=300)
        
        with st.spinner("🔍 Sedang menerawang jenis makanan..."):
            # Preprocessing
            target_size = (224, 224)
            image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            predictions = classifier_model.predict(img_batch)
            idx_max = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            # List Kelas
            CLASS_NAMES = ["bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng", 
                           "pempek", "rawon", "rendang", "sate", "soto"] 
            
            predicted_label_raw = CLASS_NAMES[idx_max]
            display_label = predicted_label_raw.replace("_", " ").title()
            
            st.success(f"🤖 AI yakin **{confidence:.1f}%** ini adalah: **{display_label}**")

            # Tombol Cari Resep
            if st.button(f"📖 Cari Resep '{display_label}' di Buku"):
                cari_resep_di_buku(display_label)

# --- MODE 2: TEKS ---
elif input_mode == "✍️ Tulis Nama Makanan":
    st.info("Masukkan nama makanan, contoh: Nasi Goreng, Rendang, atau Sambal.")
    
    user_text_input = st.text_input("Nama Makanan:")

    if user_text_input:
        makanan_dicari = user_text_input.strip().title()
        
        if st.button(f"📖 Cari Resep '{makanan_dicari}'"):
            cari_resep_di_buku(makanan_dicari)