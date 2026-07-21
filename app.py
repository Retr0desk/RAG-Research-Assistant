
import streamlit as st
import os
from dotenv import load_dotenv
from getpass import getpass
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# --- KONFIGURASI HALAMAN ---
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
st.set_page_config(page_title="RAG Document", page_icon="🤖")
st.title("RAG Research Assistant")
st.markdown("Tanyakan apapun tentang dokumen yang telah di-index di database.")

# --- KONFIGURASI BACKEND ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
load_dotenv()

# Pastikan API Token tersedia
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass("Masukkan HuggingFace API Token Anda: ")

@st.cache_resource
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="paper_collection"
    )
    
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        temperature=0.4,
        max_new_tokens=1024,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    model = ChatHuggingFace(llm=llm)
    
    template = """<|system|>
Anda adalah asisten riset yang menjawab berdasarkan dokumen. Jika tidak ada di konteks, katakan Anda tidak tahu. Jawab dalam Bahasa Indonesia.
KONTEKS: 
{context}</s>
<|user|>
{question}</s>
<|assistant|>"""
    
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | model
    | StrOutputParser()
    )
    rag_chain_with_sources = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    return rag_chain_from_docs, rag_chain_with_sources

# Load chain
try:
    rag_chain_from_docs, rag_chain_with_sources = load_rag_chain()
except Exception as e:
    st.error(f"Gagal memuat database: {e}. Pastikan sudah menjalankan 'create_database.py'.")
    st.stop()

# Inisialisasi riwayat pesan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan pesan dari riwayat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input user
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari di dokumen..."):
            # Jalankan chain (sekarang mengembalikan dict berisi 'answer' dan 'context')
            result = rag_chain_with_sources.invoke(prompt)
            answer = result["answer"]
            source_documents = result["context"]

            st.markdown(answer)

            # Tampilkan Sumber Dokumen dalam Expander
            with st.expander("📚 Lihat Sumber Dokumen"):
                for i, doc in enumerate(source_documents):
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "-")
                    st.markdown(f"**Sumber {i+1}:** {source_name} (Halaman {page})")
                    st.caption(f"{doc.page_content[:200]}...") # Cuplikan teks
    
    st.session_state.messages.append({"role": "assistant", "content": answer})