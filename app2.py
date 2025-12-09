# --- MANTRA DATABASE PINTAR (SAFE MODE) ---
import sys
import os

# Kita bungkus pakai try-except.
# Di Laptop (Windows) ini akan diskip (karena gak ada pysqlite3).
# Di Streamlit Cloud (Linux) ini akan jalan (nanti kita atur config khususnya).
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------------------------------

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import gdown

# Import LangChain
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Chef AI Mustikarasa", page_icon="🥘", layout="centered")

st.title("🥘 Detektif Resep Mustikarasa")
st.write("Temukan resep legendaris dari foto atau nama makanannya langsung!")
st.markdown("---")

# --- SETUP API KEY ---
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
# 2. SETUP MODEL & DATABASE
# ==========================================

def build_vgg_architecture():
    base_model = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax') 
    ])
    return model

@st.cache_resource
def load_image_model():
    local_model_path = "Indonesian_Food_VGG16.keras"
    if not os.path.exists(local_model_path):
        file_id = "1Odf--QVAO-q2REy0EdKiqxMU3ufRps4Y" 
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, local_model_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal download model: {e}")
            st.stop()

    try:
        model = build_vgg_architecture()
        model.predict(np.zeros((1, 224, 224, 3)), verbose=0) 
        model.load_weights(local_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_rag_system():
    # Embedding (HARUS SAMA dengan Colab)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # --- AUTO-DETECT DATABASE PATH ---
    # Kadang pas unzip, foldernya jadi dobel (chroma_db/chroma_db/...)
    # Kode ini akan mencari di mana file .sqlite3 yang asli berada.
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        st.error("❌ Folder 'chroma_db' GAK KETEMU! Cek folder projectmu.")
        st.stop()

    # Cek apakah isinya ada file sqlite3
    if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
        # Coba cek sedalam satu level (siapa tau dobel folder)
        nested_path = os.path.join(db_path, "chroma_db")
        if os.path.exists(os.path.join(nested_path, "chroma.sqlite3")):
            db_path = nested_path # Ketemu di dalam folder anak
        else:
            st.warning("⚠️ File database (chroma.sqlite3) tidak ditemukan di folder chroma_db.")
            st.write("Isi folder chroma_db kamu: ", os.listdir("./chroma_db"))

    # Load Database
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Model Gemini (Pakai 1.5 Flash biar stabil & hemat kuota)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)
    
    # MultiQuery
    output_parser_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Kamu adalah asisten chef AI. Tugasmu adalah membuat 3 variasi pertanyaan pencarian
    berdasarkan pertanyaan user, agar kita bisa menemukan resep yang tepat di database buku resep Indonesia.
    Dan kalau yang ditanyakan bukan soal resep, kasuh yang berhubungan dengan isi yang ada di buku mustika rasa ini.

    Pertanyaan User: {question}

    Keluarkan 3 variasi pertanyaan (satu per baris):"""
    )
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=output_parser_prompt
    )
    
    # Chain
    template = """Kamu adalah asisten ahli masakan Indonesia dari buku legendaris Mustikarasa.
    Gunakan potongan konteks resep di bawah ini untuk menjawab pertanyaan pengguna.
    Jika pertanyaannya bukan soal resep, berikan output yang sesuai dengan isi buku mustika rasa. 
    Kalau pertanyaanya ada nama daerahnya misal rendang padang tapi gak ada spesifik rendang padang, cari aja resep makananan yang berhubungan dengan rendang atau padangnya
    Kalau pertanyaanya bukan soal resep, kasih informasi yang berhubungan dengan buku mustika rasa. Bisa jadi daerah, bisa jadi bahan makanan yang dia punya, bisa jadi dia punya bahan tapi tidak tau cara memasaknya. 
    Intinya jawab pertanyaan sesuai kata per kata yang user input, kalau benar benar tidak ada baru anda jawab tidak ada. 

    KONTEKS RESEP:
    {context}

    PERTANYAAN USER:
    {question}

    INSTRUKSI:
    1. Jawablah dengan ramah dan lengkap.
    2. Jika ada bahan dan takaran, sebutkan dengan jelas.
    3. Jangan mengarang resep sendiri! Gunakan HANYA informasi dari konteks di atas.
    4. Jika resep tidak ditemukan di konteks, katakan jujur bahwa tidak ada di buku ini.

    JAWABAN:"""
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever_from_llm

with st.spinner("Sedang menyiapkan dapur AI..."):
    try:
        classifier_model = load_image_model()
        rag_chain, rag_retriever = load_rag_system()
        st.success("✅ Sistem Siap!")
    except Exception as e:
        st.error(f"❌ Error Loading: {e}")
        st.stop()

# ==========================================
# 3. FUNGSI PENCARI RESEP + DEBUGGER
# ==========================================
def cari_resep_di_buku(nama_makanan):
    st.markdown("---")
    st.subheader(f"📜 Hasil Pencarian: {nama_makanan}")
    
    with st.spinner(f"Chef sedang meracik resep '{nama_makanan}'..."):
        try:
            # 1. Jalankan Chain Utama
            query = f"Jelaskan resep lengkap dan cara membuat {nama_makanan}."
            hasil_resep = rag_chain.invoke(query)
            st.markdown(hasil_resep)
            
            # 2. DEBUGGING (PENTING)
            with st.expander("🛠️ KLIK SINI: Cek Apakah Database Terbaca?"):
                st.info("Mengecek isi database...")
                docs = rag_retriever.invoke(query)
                
                if len(docs) == 0:
                    st.error("⚠️ DATABASE KOSONG / TIDAK TERBACA! (0 Dokumen)")
                    st.write("Kemungkinan: Masalah versi SQLite di Laptop atau Path salah.")
                else:
                    st.success(f"✅ Database Sehat! Ditemukan {len(docs)} potongan teks.")
                    for i, doc in enumerate(docs):
                        st.caption(f"**Sumber {i+1}:** {doc.page_content[:200]}...")
                        st.divider()

        except Exception as e:
            st.error(f"Gagal mencari resep: {e}")

# ==========================================
# 4. INTERFACE UTAMA
# ==========================================
MODE_FOTO = "🖼️ Upload Foto"
MODE_TEKS = "✍️ Tulis Nama atau Resep yang ingin anda ketahui"

input_mode = st.radio(
    "Pilih cara pencarian:",
    options=[MODE_FOTO, MODE_TEKS],
    horizontal=True
)
st.markdown("---")

if input_mode == MODE_FOTO:
    uploaded_file = st.file_uploader("Upload foto makanan", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Foto Upload', width=300)
        
        with st.spinner("🔍 Memprediksi..."):
            target_size = (224, 224)
            img_array = np.asarray(ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            predictions = classifier_model.predict(img_batch)
            idx_max = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            CLASS_NAMES = ["bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng", 
                           "pempek", "rawon", "rendang", "sate", "soto"] 
            
            display_label = CLASS_NAMES[idx_max].replace("_", " ").title()
            st.info(f"🤖 Prediksi: **{display_label}** ({confidence:.1f}%)")

            if st.button(f"📖 Cari Resep '{display_label}'"):
                cari_resep_di_buku(display_label)

elif input_mode == MODE_TEKS:
    user_text_input = st.text_input("Nama Makanan (Contoh: Rendang):")
    if user_text_input:
        makanan_dicari = user_text_input.strip().title()
        if st.button(f"📖 Cari Resep '{makanan_dicari}'"):
            cari_resep_di_buku(makanan_dicari)