# index.py
import pdfplumber
import pickle
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Any
import re

class TemplateProcessor:
    def __init__(self, template_dir: str, api_key: str):
        self.template_dir = template_dir
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.metadata = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from {pdf_path}: {str(e)}")

    def process_template_answers(self) -> None:
        """Processes all template PDFs and creates FAISS index."""
        all_texts = []
        
        # Process each template PDF
        for pdf_file in os.listdir(self.template_dir):
            if pdf_file.endswith('.pdf'):
                question_number = pdf_file.split('.')[0].upper()  # e.g., "1A"
                pdf_path = os.path.join(self.template_dir, pdf_file)
                
                # Extract text and store metadata
                template_text = self.extract_text_from_pdf(pdf_path)
                max_marks = self._get_max_marks(question_number)
                
                self.metadata[question_number] = {
                    "template_text": template_text,
                    "max_marks": max_marks
                }
                
                all_texts.append({
                    "text": template_text,
                    "metadata": {"question": question_number}
                })
        
        # Create FAISS index
        vector_store = FAISS.from_texts(
            texts=[item["text"] for item in all_texts],
            embedding=self.embeddings,
            metadatas=[item["metadata"] for item in all_texts]
        )
        
        # Save index and metadata
        vector_store.save_local("template_index")
        with open("metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
    def _get_max_marks(self, question_number: str) -> int:
        """Define max marks for each question."""
        marks_mapping = {
            "1A": 5, "1B": 3, "2A": 4, "2B": 3,
            # Add more mappings as needed
        }
        return marks_mapping.get(question_number, 0)

