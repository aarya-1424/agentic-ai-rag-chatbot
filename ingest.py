from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

PDF_PATH = "data/Agentic_AI_Ebook.pdf"
DB_PATH = "vectorstore"

def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"🔢 Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Saving to FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_pdf()
