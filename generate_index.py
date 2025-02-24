import os
import pdfplumber
import pickle
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added import
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
API_KEY = "" #add google api key

# Dataset Paths
DATASET_COMPLETE = "dataset_complete"  # Folder for complete template answer sheet
DATASET_INDIVIDUAL = "dataset_individual"  # Folder for individual question answer PDFs

# Index Names
INDEX_COMPLETE = "index_complete"  # Index files for Feature 1
INDEX_INDIVIDUAL = "index_individual"  # Index files for Feature 2

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType
    return text.strip()

def generate_index_complete(api_key):
    """Generates FAISS index for the complete template answer sheet."""
    print("🔄 Extracting text from complete template answer sheet...")
    template_text = ""
    for file in os.listdir(DATASET_COMPLETE):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATASET_COMPLETE, file)
            template_text += extract_text_from_pdf(file_path)
    
    if not template_text:
        print("❌ No valid text found in the complete template PDF.")
        return

    print("📖 Splitting text into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(template_text)

    print("🔍 Generating embeddings using Google AI...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    print("📁 Creating FAISS vector store...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save FAISS index
    print("💾 Saving FAISS index...")
    vector_store.save_local(INDEX_COMPLETE)

    # Save metadata separately
    with open(f"{INDEX_COMPLETE}.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    print("✅ FAISS index and metadata saved successfully for complete template!")

def generate_index_individual(api_key):
    """Generates FAISS index for individual question answer PDFs."""
    print("🔄 Extracting template answers...")
    template_answers = {}
    
    for file in os.listdir(DATASET_INDIVIDUAL):
        if file.endswith(".pdf"):
            question_number = file.replace(".pdf", "").upper()  # Extract question ID (e.g., 1A)
            file_path = os.path.join(DATASET_INDIVIDUAL, file)
            extracted_text = extract_text_from_pdf(file_path)
            if extracted_text:
                template_answers[question_number] = extracted_text
    
    if not template_answers:
        print("❌ No valid template answers found.")
        return

    print("🔍 Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    texts = list(template_answers.values())
    question_numbers = list(template_answers.keys())

    text_embeddings = np.array([embeddings.embed_query(text) for text in texts]).astype('float32')

    print("📁 Creating FAISS index...")
    dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(text_embeddings)

    print("💾 Saving FAISS index...")
    faiss.write_index(index, f"{INDEX_INDIVIDUAL}.faiss")

    with open(f"{INDEX_INDIVIDUAL}.pkl", "wb") as f:
        pickle.dump(question_numbers, f)

    print("✅ Indexing complete for individual question answers!")

if __name__ == "__main__":
    # Generate index for complete template answer sheet
    generate_index_complete(API_KEY)

    # Generate index for individual question answer PDFs
    generate_index_individual(API_KEY)