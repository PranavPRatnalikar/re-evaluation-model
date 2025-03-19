# 📄 Automated Answer Evaluation System

A powerful tool that leverages **FAISS**, **Google Generative AI Embeddings**, and **Streamlit** to evaluate student answers by comparing them with a template answer sheet.

This system provides **two key features**:
1. **Complete Template Answer Sheet Comparison** - Evaluates a student’s entire answer sheet against a full reference sheet.
2. **Individual Question Answer PDFs** - Compares student responses on a per-question basis.

---

## 🚀 Features

- 📂 **Extracts text from uploaded PDF answer sheets**  
- 🔍 **Compares student responses with template answers using AI-based embeddings**  
- 📊 **Computes similarity scores and assigns marks automatically**  
- 📜 **Supports bulk processing of answer sheets**  
- 🎯 **Uses FAISS for fast retrieval and efficient answer evaluation**  

---

## 🛠️ Installation & Setup

Follow these steps to set up the project on your local system.

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/PranavPRatnalikar/re-evaluation-model.git
cd re-evaluation-model
```

### 2️⃣ **Create a Virtual Environment (Optional but Recommended)**
```sh
python -m venv venv
source venv/bin/activate  # For MacOS/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4️⃣ **Set Up Your API Key**
- Obtain a **Google Generative AI API Key** from [Google AI](https://ai.google.com).
- Add the API key in `app.py` and `generate_index.py` where indicated:
```python
API_KEY = "your_google_api_key_here"
```

### 5️⃣ **Prepare Data**
- **Complete Template Answer Sheet PDFs** should be placed in the `dataset_complete/` folder.
- **Individual Question Answer PDFs** should be placed in the `dataset_individual/` folder.

### 6️⃣ **Generate FAISS Index**
Run the following command to create FAISS indexes for efficient answer retrieval:
```sh
python generate_index.py
```

### 7️⃣ **Run the Streamlit App**
Start the **Streamlit UI**:
```sh
streamlit run app.py
```
The app will be available at:
```
http://localhost:8501
```

---

## 📁 Folder Structure

```
re-evaluation-model/
│── dataset_complete/        # Store the complete answer template PDFs here
│── dataset_individual/      # Store individual question-answer PDFs here
│── app.py                   # Streamlit web app
│── generate_index.py         # FAISS index generation script
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---


## 🎯 Technologies Used

- 🏗 **[FAISS](w)** - Fast retrieval of embeddings for similarity search  
- 🤖 **[Google Generative AI](w)** - Embeddings for document comparison  
- 🖥 **[Streamlit](w)** - Interactive web UI for evaluation  
- 📚 **[pdfplumber](w)** - Extracts text from PDFs  
- 🐍 **Python 3.x** - Core language  

