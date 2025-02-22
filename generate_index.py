import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

# Configuration
TEMPLATE_DIR = "dataset"  # Folder containing template answer sheet PDFs
INDEX_NAME = "index"       # Prefix for FAISS index files
API_KEY = "AIzaSyArdn9_Uabo9q0aYmm4dxybVEb0tj7dlrk"  

def extract_text_from_pdfs(pdf_folder):
    """Extracts text from all PDFs in the given folder."""
    text = ""
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            with pdfplumber.open(file_path) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # Handle NoneType
    return text.strip()

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into smaller chunks for better embedding processing."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def generate_faiss_index(api_key):
    """Generates FAISS index from template answer sheet text."""
    print("🔄 Extracting text from template answer sheet...")
    template_text = extract_text_from_pdfs(TEMPLATE_DIR)

    if not template_text:
        print("❌ No valid text found in template PDFs.")
        return

    print("📖 Splitting text into smaller chunks...")
    text_chunks = split_text_into_chunks(template_text)

    print("🔍 Generating embeddings using Google AI...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    print("📁 Creating FAISS vector store...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save FAISS index
    print("💾 Saving FAISS index...")
    vector_store.save_local(INDEX_NAME)

    # Save metadata separately
    with open(f"{INDEX_NAME}.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    print("✅ FAISS index and metadata saved successfully!")

if __name__ == "__main__":
    generate_faiss_index(API_KEY)
