import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

# Import LangChain Standard
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- TAMBAHAN PENTING (MultiQuery) ---
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import logging

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
# 2. SETUP MODEL & DATABASE (CACHE)
# ==========================================

# --- FUNGSI BANGUN MODEL MANUAL (VGG16) ---
def build_vgg_architecture():
    base_model = tf.keras.applications.VGG16(
        include_top=False, weights=None, input_shape=(224, 224, 3))
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
        print(f"Sedang mendownload model ke {local_model_path}...")
        try:
            gdown.download(url, local_model_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal download model: {e}")
            st.stop()

    try:
        model = build_vgg_architecture()
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
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. Setup Otak (PAKE 1.5 FLASH BIAR KUOTA AMAN)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    
    # --- 4. MULTI-QUERY RETRIEVER (LOGIKA PINTAR) ---
    # Prompt khusus biar LLM mikirin variasi pertanyaan
    output_parser_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Kamu adalah asisten chef AI. Tugasmu adalah membuat 3 variasi pertanyaan pencarian
    berdasarkan pertanyaan user, agar kita bisa menemukan resep yang tepat di database buku resep Indonesia.
    Dan kalau yang ditanyakan bukan soal resep, kasuh yang berhubungan dengan isi yang ada di buku mustika rasa ini.

    Pertanyaan User: {question}

    Keluarkan 3 variasi pertanyaan (satu per baris):
        """
    )

    # Gabungkan Retrieval dengan Kecerdasan LLM
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=output_parser_prompt
    )
    
    # --- 5. RAG CHAIN ---
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

    JAWABAN:
    """
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- LOAD SISTEM ---
with st.spinner("Sedang menyiapkan dapur AI..."):
    try:
        classifier_model = load_image_model()
        rag_chain = load_rag_system()
        st.success("✅ Sistem Siap!")
    except Exception as e:
        st.error(f"❌ Error Loading: {e}")
        st.stop()

# ==========================================
# 3. FUNGSI PENCARI RESEP
# ==========================================
def cari_resep_di_buku(nama_makanan):
    st.markdown("---")
    st.subheader(f"📜 Hasil Pencarian: {nama_makanan}")
    with st.spinner(f"Sedang mencari berbagai variasi resep '{nama_makanan}' di buku..."):
        try:
            query = f"Bagaimana cara memasak {nama_makanan}? Sebutkan bahan dan langkahnya."
            hasil_resep = rag_chain.invoke(query)
            st.markdown(hasil_resep)
        except Exception as e:
            st.error(f"Gagal mencari resep (Kuota Habis/Error): {e}")

# ==========================================
# 4. INTERFACE UTAMA
# ==========================================
input_mode = st.radio(
    "Pilih cara pencarian:",
    ("🖼️ Upload Foto", "✍️ Tulis Nama atau Resep yang ingin anda ketahui"),
    horizontal=True
)
st.markdown("---")

# --- MODE GAMBAR ---
if input_mode == "🖼️ Upload Foto":
    uploaded_file = st.file_uploader("Upload foto makanan (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Foto yang diupload', width=300)
        
        with st.spinner("🔍 VGG16 sedang memprediksi..."):
            target_size = (224, 224)
            image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            predictions = classifier_model.predict(img_batch)
            idx_max = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            
            CLASS_NAMES = ["bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng", 
                           "pempek", "rawon", "rendang", "sate", "soto"] 
            
            predicted_label_raw = CLASS_NAMES[idx_max]
            display_label = predicted_label_raw.replace("_", " ").title()
            
            st.info(f"🤖 AI memprediksi: **{display_label}** ({confidence:.1f}%)")

            if st.button(f"📖 Cari Resep '{display_label}'"):
                cari_resep_di_buku(display_label)

# --- MODE TEKS ---
elif input_mode == "✍️ Tulis Nama atau Resep yang ingin anda ketahui":
    user_text_input = st.text_input("Masukkan nama makanan (Misal: Nasi Liwet):")
    if user_text_input:
        makanan_dicari = user_text_input.strip().title()
        if st.button(f"📖 Cari Resep '{makanan_dicari}'"):
            cari_resep_di_buku(makanan_dicari) 