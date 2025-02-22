import streamlit as st
import json
from typing import Dict, Tuple, Any

class AnswerEvaluator:
    def __init__(self, api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.vector_store = FAISS.load_local("template_index", self.embeddings)
        with open("metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
            
    def extract_student_answers(self, pdf_file) -> Dict[str, str]:
        """Extracts answers from student PDF using question markers."""
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                
        # Extract answers using regex
        answers = {}
        pattern = r"###(\w+)(.*?)(?=###\w+|$)"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            question_num = match.group(1).upper()
            answer_text = match.group(2).strip()
            answers[question_num] = answer_text
            
        return answers
    
    def compute_similarity(self, student_answer: str, question_num: str) -> float:
        """Computes similarity between student answer and template."""
        if not student_answer:
            return 0.0
            
        # Get most similar template answer
        results = self.vector_store.similarity_search_with_score(
            student_answer,
            k=1,
            filter={"question": question_num}
        )
        
        if not results:
            return 0.0
            
        # Convert distance to similarity score (0-1)
        similarity = 1 / (1 + results[0][1])
        return similarity
    
    def calculate_marks(self, similarity: float, max_marks: int) -> float:
        """Calculates marks based on similarity thresholds."""
        if similarity >= 0.8:
            return max_marks
        elif 0.6 <= similarity < 0.8:
            return max_marks * 0.5
        return 0
    
    def evaluate_submission(self, pdf_file) -> Dict[str, Any]:
        """Evaluates complete student submission."""
        # Extract all answers
        student_answers = self.extract_student_answers(pdf_file)
        results = {}
        
        # Evaluate each question
        for question_num, metadata in self.metadata.items():
            student_answer = student_answers.get(question_num, "")
            
            # Check for duplicate answers
            if self._is_duplicate_answer(student_answer, student_answers):
                similarity = 0.0
            else:
                similarity = self.compute_similarity(student_answer, question_num)
            
            marks = self.calculate_marks(similarity, metadata["max_marks"])
            
            results[question_num] = {
                "similarity": round(similarity, 2),
                "marks_obtained": marks,
                "max_marks": metadata["max_marks"]
            }
            
        return results
    
    def _is_duplicate_answer(self, answer: str, all_answers: Dict[str, str]) -> bool:
        """Checks for duplicate answers."""
        if not answer:
            return False
        normalized = self._normalize_text(answer)
        count = sum(1 for other in all_answers.values() 
                   if self._normalize_text(other) == normalized)
        return count > 1
    
    def _normalize_text(self, text: str) -> str:
        """Normalizes text for comparison."""
        return ' '.join(text.lower().split())

# Streamlit UI
st.title("📝 Answer Evaluation System")

# API Key Input
api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
if not api_key:
    st.warning("Please enter your Google API Key.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload Student Answer Sheet (PDF)", type="pdf")

if uploaded_file:
    try:
        evaluator = AnswerEvaluator(api_key)
        
        with st.spinner("Evaluating submission..."):
            results = evaluator.evaluate_submission(uploaded_file)
        
        # Display Results
        st.header("Evaluation Results")
        
        total_marks = 0
        max_total = 0
        
        for question_num, data in results.items():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Question {question_num}")
                st.write(f"Similarity: {data['similarity']:.2%}")
                st.write(f"Marks: {data['marks_obtained']}/{data['max_marks']}")
            
            with col2:
                st.progress(data['similarity'])
            
            total_marks += data['marks_obtained']
            max_total += data['max_marks']
        
        # Show Final Score
        st.header("Final Score")
        st.write(f"Total Marks: {total_marks}/{max_total}")
        st.write(f"Percentage: {(total_marks/max_total)*100:.2f}%")
        
        # Download Results
        st.download_button(
            "Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="evaluation_results.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error processing submission: {str(e)}")

if __name__ == "__main__":
    st.sidebar.markdown("### Instructions")
    st.sidebar.write("""
    1. Enter your Google API Key
    2. Upload student's PDF file
    3. View results and download report
    """)