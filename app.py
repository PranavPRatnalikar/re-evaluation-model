import streamlit as st
import pdfplumber
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Streamlit UI
st.title("📄 Answer Similarity Checker")

# API Key Input
api_key = 'AIzaSyArdn9_Uabo9q0aYmm4dxybVEb0tj7dlrk'
# api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

# if not api_key:
#     st.warning("Please enter a valid API key.")

# Load FAISS Index
INDEX_NAME = "index"

def load_faiss_index(api_key):
    """Loads the FAISS index and precomputed text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        vector_store = FAISS.load_local(INDEX_NAME, embeddings, allow_dangerous_deserialization=True)
        with open(f"{INDEX_NAME}.pkl", "rb") as f:
            text_chunks = pickle.load(f)
        return vector_store, text_chunks
    except Exception as e:
        st.error(f"❌ Error loading FAISS index: {e}")
        return None, None

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a student answer sheet (PDF)."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType
    return text.strip()

# Compute Similarity
def compute_similarity(student_answer, vector_store):
    """Finds the most similar template answer and computes relevance score."""
    if not student_answer:
        return "No answer provided.", 0.0

    results = vector_store.similarity_search_with_score(student_answer, k=1)
    
    if not results:
        return "No relevant match found.", 0.0
    
    matched_doc, score = results[0]  # `results[0]` returns (Document, distance)
    
    # Convert FAISS distance into a similarity percentage (lower distance = higher similarity)
    similarity_percentage = round((1 / (1 + score)) * 100, 2)

    return matched_doc.page_content, similarity_percentage



# Main Code
vector_store, text_chunks = load_faiss_index(api_key)

# if vector_store:
#     st.sidebar.success("✅ FAISS index loaded successfully.")

uploaded_file = st.file_uploader("📂 Upload Student Answer Sheet (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from student answer..."):
        student_answer = extract_text_from_pdf(uploaded_file)
    
    if student_answer:
        st.text_area("📜 Extracted Student Answer:", student_answer, height=150)

        if st.button("🔍 Check Similarity"):
            with st.spinner("Comparing with template answer..."):
                matched_text, similarity_score = compute_similarity(student_answer, vector_store)
                
                st.subheader("📊 Similarity Score:")
                st.write(f"**{similarity_score}% relevant to the template answer.**")

                # st.subheader("📌 Closest Matching Template Answer:")
                # st.info(matched_text)
