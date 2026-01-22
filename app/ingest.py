import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

DATA_DIR = "data/documents"
FAISS_PATH = "embeddings/faiss_index"

def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def create_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)

def ingest_documents():
    docs = load_documents()
    chunks = split_documents(docs)
    create_faiss_index(chunks)

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)

    print("Creating FAISS vector store...")
    create_faiss_index(chunks)

    print("✅ Ingestion completed successfully")
