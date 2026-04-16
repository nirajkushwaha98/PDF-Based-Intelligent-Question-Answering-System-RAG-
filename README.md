# 📄 PDF Question Answering System (RAG-Based)

## 🚀 Overview

This project is a PDF-based Question Answering system that allows users to upload documents and ask questions in natural language. The system extracts text from PDFs, retrieves relevant content, and generates answers using a generative AI model.

It follows a simplified Retrieval-Augmented Generation (RAG) approach.

---

## 🧠 Features

* Upload and process PDF documents
* Extract and clean text automatically
* Ask questions based on document content
* Retrieve relevant text using keyword matching
* Generate answers using AI
* Fallback response if AI fails
* Maintain conversation history

---

## ⚙️ Tech Stack

* **Backend:** Python, Flask
* **PDF Processing:** PyPDF2
* **AI Model:** Google Gemini API
* **Text Processing:** Regex, Tokenization

---

## 📂 Project Structure

```
project/
│── app.py                # Main Flask application
│── requirements.txt     # Dependencies
│── .env                 # API keys
│── README.md            # Project documentation
```

---

## 🔧 Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/pdf-qa-system.git
cd pdf-qa-system
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Setup environment variables

Create a `.env` file and add:

```
GEMINI_API_KEY=your_api_key_here
```

---

## ▶️ Running the Project

```
python app.py
```

Server will start at:

```
http://127.0.0.1:5000
```

---

## 📌 API Endpoints

### 1. Upload PDF

**POST** `/upload-pdf`

### 2. Ask Question

**POST** `/ask`

### 3. List PDFs

**GET** `/pdfs`

### 4. Clear PDF Data

**DELETE** `/clear/<pdf_id>`

---

## 🧪 Example Workflow

1. Upload a PDF
2. Get `pdf_id`
3. Ask questions using that `pdf_id`
4. Receive answers based on document

---

## ⚠️ Limitations

* Uses keyword-based retrieval (not semantic search)
* No database (in-memory storage only)
* Depends on external AI API
* Not optimized for very large PDFs

---

## 🚀 Future Improvements

* Add semantic search using embeddings
* Integrate vector database (FAISS)
* Support multiple documents
* Build frontend UI
* Add summarization feature

---

## 📖 References

* Google Gemini API Docs
* Flask Documentation
* PyPDF2 Documentation

---

## 👨‍💻 Author

Developed as an academic project for learning purposes.

---

## ⚠️ Note

This is a basic implementation of a RAG-like system and not a production-ready solution.
