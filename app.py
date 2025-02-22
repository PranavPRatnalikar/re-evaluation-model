import os
import streamlit as st
import pdfplumber
import pickle
import faiss
import numpy as np
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Configuration
API_KEY = "AIzaSyArdn9_Uabo9q0aYmm4dxybVEb0tj7dlrk"

# Streamlit UI
st.title("📄 Automated Answer Evaluation System")

# Feature Selection
feature = st.sidebar.radio("Select Feature", ["Complete Template Answer Sheet", "Individual Question Answer PDFs"])

# Load FAISS index and metadata
def load_faiss_index(feature, api_key):
    """Loads FAISS index and metadata based on the selected feature."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    if feature == "Complete Template Answer Sheet":
        index_name = "index_complete"
        try:
            vector_store = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
            with open(f"{index_name}.pkl", "rb") as f:
                text_chunks = pickle.load(f)
            return vector_store, text_chunks, embeddings
        except Exception as e:
            st.error(f"❌ Error loading FAISS index: {e}")
            return None, None, None
    else:
        index_name = "index_individual"
        try:
            index = faiss.read_index(f"{index_name}.faiss")
            with open(f"{index_name}.pkl", "rb") as f:
                question_numbers = pickle.load(f)
            return index, question_numbers, embeddings
        except Exception as e:
            st.error(f"❌ Error loading FAISS index: {e}")
            return None, None, None

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from a student answer sheet (PDF)."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf_reader:
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType
    return text.strip()

# Extract student answers for Feature 2
def extract_student_answers(pdf_file):
    """Extracts question-wise answers from student PDF."""
    text = extract_text_from_pdf(pdf_file)
    
    # Extract answers based on the ###QuestionNumber format
    answers = {}
    pattern = r"(###\d+[A-Z])\s*(.+?)(?=###|\Z)"  # Matches "###1A" followed by the answer
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, answer = match
        answers[question.upper()] = answer.strip()
    
    return answers

# Compute similarity for Feature 1
def compute_similarity_complete(student_answer, vector_store):
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

# Compute similarity for Feature 2
def compute_similarity_individual(student_answer, index, question_numbers, embeddings):
    """Finds most similar template answer and calculates similarity."""
    if not student_answer:
        return "No answer provided.", 0.0
    
    student_embedding = np.array(embeddings.embed_query(student_answer)).astype('float32').reshape(1, -1)
    _, closest_idx = index.search(student_embedding, 1)  # Retrieve nearest neighbor
    
    matched_question = question_numbers[closest_idx[0][0]]
    return matched_question, (1 / (1 + _[0][0])) * 100  # Convert L2 distance to similarity %

# Evaluate answers for Feature 2
def evaluate_answers(student_answers, index, question_numbers, embeddings, max_marks=5):
    results = {}

    for question, student_answer in student_answers.items():
        matched_question, similarity = compute_similarity_individual(student_answer, index, question_numbers, embeddings)

        # Calculate marks as a percentage of max_marks
        marks_obtained = (similarity * max_marks) / 100

        # Round marks to nearest integer or .5
        decimal_part = marks_obtained - int(marks_obtained)
        if decimal_part < 0.25:
            marks_obtained = int(marks_obtained)
        elif 0.25 <= decimal_part < 0.75:
            marks_obtained = int(marks_obtained) + 0.5
        else:
            marks_obtained = int(marks_obtained) + 1

        results[question] = {
            "similarity": f"{round(similarity, 2)}%",  # Format similarity as percentage
            "marks_obtained": marks_obtained,
            "max_marks": max_marks
        }
    
    return results

# Main Logic
index, metadata, embeddings = load_faiss_index(feature, API_KEY)

uploaded_file = st.file_uploader("📂 Upload Student Answer Sheet (PDF)", type="pdf")

if uploaded_file:
    if feature == "Complete Template Answer Sheet":
        with st.spinner("Extracting text from student answer..."):
            student_answer = extract_text_from_pdf(uploaded_file)
        
        if student_answer:
            st.text_area("📜 Extracted Student Answer:", student_answer, height=150)

            if st.button("🔍 Check Similarity"):
                with st.spinner("Comparing with template answer..."):
                    matched_text, similarity_score = compute_similarity_complete(student_answer, index)
                    
                    st.subheader("📊 Similarity Score:")
                    st.write(f"**{similarity_score}% relevant to the template answer.**")
    else:
        with st.spinner("Extracting text from student answer sheet..."):
            student_answers = extract_student_answers(uploaded_file)
        
        if student_answers:
            st.text_area("📜 Extracted Student Answers:", "\n".join(f"{q}: {a}" for q, a in student_answers.items()), height=150)

            if st.button("🔍 Evaluate Answers"):
                with st.spinner("Comparing answers with templates..."):
                    results = evaluate_answers(student_answers, index, metadata, embeddings)
                    
                    st.subheader("📊 Score Breakdown:")
                    st.json(results)
                    
                    total_marks = sum(v["marks_obtained"] for v in results.values())
                    st.subheader(f"🏆 Total Score: {total_marks} Marks")