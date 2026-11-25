"""
CV Reviewer: revisi CV agar sesuai requirement lowongan.
LangChain + Groq (Llama 3.1).

Alur:
    CV (PDF) -> teks -> (CV pendek: dikirim utuh | dokumen panjang: RAG dengan Chroma)
    + requirement lowongan + pertanyaan -> prompt -> LLM -> analisis & saran revisi
"""

import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
HISTORY_TURNS = 3  # jumlah tanya-jawab terakhir yang ikut dikirim ke model
# CV 1-3 halaman muat utuh di prompt -> lebih akurat daripada dipotong-potong.
# Dokumen lebih panjang dari ini baru memakai RAG (cari potongan yang relevan).
MAX_FULL_CONTEXT_CHARS = 12000

ANALYSIS_REQUEST = (
    "Analisis CV saya terhadap requirement lowongan ini dan berikan saran revisinya."
)

load_dotenv()


def get_api_key() -> str | None:
    """Ambil API key dari .env (lokal) atau st.secrets (Streamlit Cloud)."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key.strip()
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


# ------------------------------------------------------------------
# Resource berat dimuat SEKALI saja
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model embedding...")
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model_name=LLM_MODEL, temperature=0.2, api_key=api_key)


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Anda adalah career coach dan recruiter berpengalaman di Indonesia yang membantu
pelamar merevisi CV agar sesuai dengan requirement lowongan kerja dan lolos ATS
(Applicant Tracking System).

Aturan penting:
1. Gunakan HANYA fakta yang ada di CV. Jangan pernah mengarang pengalaman, angka,
   sertifikat, atau skill yang tidak tertulis. Kalau sebuah requirement belum terlihat
   di CV, katakan belum terlihat dan sarankan cara pelamar membuktikannya
   (misalnya membuat project kecil), bukan menulis seolah-olah sudah punya.
2. Jawab dalam bahasa Indonesia yang jelas dan langsung ke inti.
3. Contoh kalimat CV yang Anda tulis boleh dalam bahasa Inggris jika CV-nya berbahasa Inggris.
4. Jika requirement lowongan belum diisi, minta pengguna mengisinya terlebih dahulu.
5. Versi "Sesudah" hanya boleh menyusun ulang, memperjelas, atau memakai istilah yang lebih tepat
   untuk fakta yang SUDAH ada di kalimat "Sebelum". Dilarang menambah alat, angka, tautan,
   hasil, atau aktivitas baru. Jika menurut Anda sebuah detail perlu ditambahkan, tulis dalam
   kurung siku sebagai saran, misalnya: [tambahkan jumlah data jika ada].
6. Jangan menghitung ulang skor setelah revisi; cukup satu skor di awal.

Jika pengguna meminta analisis CV, gunakan format berikut:

## Skor Kecocokan: X/100
Satu atau dua kalimat alasan skor.

## Requirement yang Sudah Terpenuhi
- Requirement -> bukti dari CV (kutip singkat)

## Requirement yang Belum Terlihat di CV
- Requirement -> saran cara memenuhinya atau menonjolkannya

## Saran Revisi per Bagian
Untuk setiap bagian yang perlu diubah (Ringkasan, Pengalaman, Project, Skills, Pendidikan):
- **Sebelum:** kalimat asli dari CV
- **Sesudah:** versi revisi yang menonjolkan requirement, memakai kata kerja aktif dan hasil terukur bila angkanya ada di CV

## Kata Kunci ATS yang Perlu Ditambahkan
Daftar kata kunci dari lowongan yang belum ada di CV tetapi relevan dengan pengalaman pelamar.

## Prioritas
Tiga perubahan paling berdampak, urut dari yang terpenting.

Untuk pertanyaan lanjutan (misalnya menulis ulang satu bagian, membuat ringkasan,
atau membuat email lamaran), jawab langsung sesuai permintaan tanpa mengulang analisis lengkap.""",
        ),
        (
            "human",
            """Requirement lowongan:
{requirements}

Isi CV:
{context}

Riwayat percakapan:
{history}

Permintaan: {question}""",
        ),
    ]
)


# ------------------------------------------------------------------
# Pemrosesan dokumen
# ------------------------------------------------------------------
def load_pdfs(uploaded_files) -> list:
    """Baca semua PDF menjadi daftar Document (satu per halaman)."""
    docs = []
    for uploaded in uploaded_files:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            for doc in PyPDFLoader(tmp_path).load():
                doc.metadata["source"] = uploaded.name
                docs.append(doc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return [d for d in docs if d.page_content.strip()]


def build_vectorstore(docs) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return Chroma.from_documents(
        documents=splitter.split_documents(docs),
        embedding=load_embeddings(),
        collection_name=f"docs-{uuid.uuid4().hex}",
    )


def page_label(doc) -> str:
    return f"{doc.metadata.get('source', 'dokumen')} - halaman {doc.metadata.get('page', 0) + 1}"


def format_docs(docs) -> str:
    return "\n\n".join(f"[{page_label(d)}]\n{d.page_content}" for d in docs)


def unique_sources(docs) -> list[str]:
    seen = []
    for d in docs:
        if page_label(d) not in seen:
            seen.append(page_label(d))
    return seen


def format_history(messages) -> str:
    recent = messages[-HISTORY_TURNS * 2:]
    if not recent:
        return "(belum ada)"
    return "\n".join(
        f"{'Pengguna' if m['role'] == 'user' else 'Asisten'}: {m['content']}" for m in recent
    )


def get_context(question: str) -> tuple[str, list[str], str]:
    """
    Pilih cara mengirim CV ke model:
    - pendek  -> seluruh teks (tidak ada bagian yang terlewat)
    - panjang -> RAG: hanya potongan paling relevan dengan pertanyaan + requirement
    """
    docs = st.session_state.docs
    full_text = format_docs(docs)
    if len(full_text) <= MAX_FULL_CONTEXT_CHARS:
        return full_text, unique_sources(docs), "CV dibaca utuh"

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = build_vectorstore(docs)
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 4, "lambda_mult": 0.7},
    )
    query = f"{question}\n{st.session_state.requirements}"
    found = retriever.invoke(query)
    return format_docs(found), unique_sources(found), "Potongan relevan (RAG)"


def reset_all():
    st.session_state.docs = []
    st.session_state.vectorstore = None
    st.session_state.file_key = None
    st.session_state.messages = []
    st.session_state.requirements = ""
    st.session_state.uploader_key += 1


def show_sources(sources, mode):
    if sources:
        with st.expander(f"Sumber ({mode})"):
            for src in sources:
                st.markdown(f"- {src}")


# ------------------------------------------------------------------
# Tampilan
# ------------------------------------------------------------------
st.set_page_config(page_title="CV Reviewer", page_icon="📄")
st.title("📄 CV Reviewer")
st.caption(f"Revisi CV sesuai requirement lowongan · LangChain + {LLM_MODEL} (Groq)")

for key, default in {
    "docs": [],
    "vectorstore": None,
    "file_key": None,
    "messages": [],
    "requirements": "",
    "uploader_key": 0,
}.items():
    st.session_state.setdefault(key, default)

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY belum diatur. Isi di file .env (lokal) atau di Secrets (Streamlit Cloud).")
    st.stop()

with st.sidebar:
    st.header("1. Unggah CV")
    uploaded_files = st.file_uploader(
        "CV dalam format PDF",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader-{st.session_state.uploader_key}",
    )

    if uploaded_files:
        file_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
        if file_key != st.session_state.file_key:
            with st.spinner("Membaca CV..."):
                try:
                    docs = load_pdfs(uploaded_files)
                except Exception as exc:
                    docs = []
                    st.error(f"Gagal membaca PDF: {exc}")
            st.session_state.file_key = file_key
            st.session_state.docs = docs
            st.session_state.vectorstore = None
            st.session_state.messages = []
            if docs:
                st.success(f"CV terbaca ({len(docs)} halaman).")
            else:
                st.warning("Tidak ada teks yang bisa dibaca. CV hasil scan (gambar) belum didukung.")

    st.header("2. Requirement lowongan")
    st.text_area(
        "Tempel kualifikasi dan deskripsi pekerjaan",
        key="requirements",
        height=220,
        placeholder="Contoh:\n- Punya pengalaman dengan Python atau JavaScript\n- Pernah pakai LLM API (OpenAI, Gemini, LangChain)\n- Punya portfolio atau GitHub",
    )

    analyze = st.button(
        "🔍 Analisis CV",
        type="primary",
        use_container_width=True,
        disabled=not (st.session_state.docs and st.session_state.requirements.strip()),
    )
    st.button("🔄 Reset", on_click=reset_all, use_container_width=True)

# Riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"), message.get("mode", ""))

if not st.session_state.docs:
    st.info("👈 Unggah CV (PDF) dan tempel requirement lowongan di sidebar, lalu klik **Analisis CV**.")
    st.stop()

if not st.session_state.requirements.strip():
    st.info("👈 Tempel requirement lowongan di sidebar supaya saran revisinya tepat sasaran.")

typed = st.chat_input("Contoh: tulis ulang ringkasan CV saya untuk posisi ini")
question = ANALYSIS_REQUEST if analyze else typed

if question:
    history = format_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Membaca CV dan requirement..."):
            try:
                context, sources, mode = get_context(question)
                chain = PROMPT | load_llm(api_key) | StrOutputParser()
                answer = chain.invoke(
                    {
                        "requirements": st.session_state.requirements.strip() or "(belum diisi)",
                        "context": context,
                        "history": history,
                        "question": question,
                    }
                )
            except Exception as exc:
                answer = f"Maaf, terjadi kesalahan saat menghubungi model: {exc}"
                sources, mode = [], ""

        st.markdown(answer)
        show_sources(sources, mode)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "mode": mode}
    )