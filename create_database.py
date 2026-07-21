import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- KONFIGURASI ---
DATA_PATH = "./data"
PDF_PATH = os.path.join(DATA_PATH, "pdf")
MD_PATH = os.path.join(DATA_PATH, "md")
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"--- Inisialisasi Model Embedding: {EMBEDDING_MODEL} ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    documents = load_documents()
    chunks = split_documents(documents)
    print(f"--- Menyimpan ke Database di: {CHROMA_PATH} ---")
    save_to_chroma(chunks, embeddings)
    print("\n✅ Database Berhasil Dibuat!")

def load_documents():
    pdf_loader = DirectoryLoader(
        PDF_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    # Loader untuk Markdown
    md_loader = DirectoryLoader(
        MD_PATH, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader
    )
    pdf_docs = pdf_loader.load()
    md_docs = md_loader.load()
    docs = pdf_docs + md_docs
    print(f"Berhasil memuat {len(pdf_docs)} halaman PDF dan {len(md_docs)} file Markdown.")
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks, embeddings):
    if os.path.exists(CHROMA_PATH):
        print(f"Menghapus database lama di {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH,
        collection_name="paper_collection"
    )
    print(f"Berhasil menyimpan {len(chunks)} chunks ke database.")

if __name__ == "__main__":
    main()