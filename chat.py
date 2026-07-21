import os
import getpass
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURASI ---
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
load_dotenv()

def main():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Masukkan HuggingFace API Token Anda: ")
    print("--- Menghubungkan ke Database ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Database tidak ditemukan di {CHROMA_PATH}. Jalankan 'create_database.py' dulu!")
        return
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
Anda adalah asisten riset yang membantu menjawab pertanyaan berdasarkan konteks dokumen yang diberikan. 
Jika jawaban tidak ada dalam konteks, katakan bahwa Anda tidak mengetahuinya.
Jawablah dalam Bahasa Indonesia yang sopan.

KONTEKS:
{context}</s>
<|user|>
{question}</s>
<|assistant|>"""

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    print("\n" + "="*50)
    print("🤖 AI AGENT READY! (Ketik 'keluar' untuk berhenti)")
    print("="*50 + "\n")

    while True:
        user_input = input("Pertanyaan Anda: ")
        
        if user_input.lower() in ['exit', 'keluar', 'quit']:
            print("Terima kasih, sampai jumpa!")
            break
        
        if not user_input.strip():
            continue

        print("\nSearching documents...", end="\r")
        
        try:
            response = rag_chain.invoke(user_input)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"\nTerjadi kesalahan: {e}\n")

if __name__ == "__main__":
    main()