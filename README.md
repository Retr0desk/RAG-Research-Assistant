# 📚 RAG Research Assistant

Asisten **Retrieval-Augmented Generation (RAG)** cerdas yang dibangun menggunakan **LangChain**, **ChromaDB**, dan **Hugging Face**. Proyek ini memungkinkan Anda memasukkan dokumen lokal (PDF, Markdown, Teks) ke dalam *vector database* lokal untuk melakukan pencarian semantik serta tanya-jawab menggunakan model *Large Language Model* (LLM) seperti Zephyr-7B.

---

## ✨ Fitur Utama

* **Ingesti Dokumen Multi-Format**: Memproses file PDF, Markdown, dan teks menggunakan `DirectoryLoader` serta *recursive character text splitter*.
* **Penyimpanan Vektor Lokal**: Pencarian kemiripan (*similarity search*) yang cepat dan tersimpan secara lokal menggunakan **ChromaDB**.
* **Hugging Face LLM & Embeddings**:
  * **LLM**: `HuggingFaceH4/zephyr-7b-beta` via Hugging Face Inference API.
  * **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
* **Arsitektur Agen**: Eksperimen agen yang fleksibel menggunakan sintaks LangChain LCEL untuk penanganan *prompt* dinamis.
* **Keamanan Variabel Lingkungan**: Pengelolaan API Token yang aman menggunakan `python-dotenv`.

---

## 📁 Struktur Repository

```text
RAG/
├── data/                 # Dokumen mentah (PDF, MD, CSV) untuk basis pengetahuan
├── chroma_db/            # Database vektor lokal ChromaDB (Diabaikan oleh Git)
├── .env                  # File kunci API dan rahasia lokal (Diabaikan oleh Git)
├── .env.example          # Template variabel lingkungan
├── .gitignore            # Daftar file/folder yang diabaikan oleh Git
├── chat.py               # Interface CLI untuk tanya-jawab dokumen
├── create_database.py    # Script untuk memotong dokumen & membangun indeks ChromaDB
└── README.md             # Dokumentasi proyek

1. Prasyarat
Pastikan Anda sudah menginstal Python 3.10+ di sistem Anda.

2. Clone Repository
git clone [https://github.com/Retr0desk/RAG-Research-Assistant.git](https://github.com/Retr0desk/RAG-Research-Assistant.git)
cd RAG-Research-Assistant

3. Buat & Aktifkan Virtual Environment
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

4. Install Dependensi
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-text-splitters python-dotenv bs4 pypdf

5. Konfigurasi Environment Variable
Salin file .env.example menjadi .env:
cp .env.example .env


💡 Cara Penggunaan
1. Ingesti Dokumen
Simpan file dokumen/riset Anda di dalam folder data/, lalu jalankan script pembuatan database vektor:
python create_database.py

2. Jalankan Chatbot Tanya-Jawab
Mulai sesi interaktif tanya-jawab berdasarkan dokumen yang sudah di-ingest:
python chat.py

3. Eksperimen Agen
Untuk menjalankan alur RAG berbasis agen dengan dynamic prompt:
python app.py
