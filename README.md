# 🤖 RAG-Based PDF Question Answering Chatbot

An AI-powered PDF Question Answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to ask questions about document content and receive context-based answers using semantic search and a Large Language Model (LLM).

🌐 **Live Demo:** https://dojsepkvptduhemt2u6myf.streamlit.app/

## 📌 Project Overview

This project demonstrates how Retrieval-Augmented Generation (RAG) works in a real-world AI application.

Instead of asking an LLM to answer questions using only its pretrained knowledge, this application retrieves relevant information from a PDF document and provides that information as context to the LLM.

The application uses:
- Sentence Transformers for generating text embeddings.
- FAISS for efficient similarity search.
- Groq API for LLM-based answer generation.
- Streamlit for the interactive web interface.

## ✨ Features

- **PDF Processing:** Extracts text from PDF documents.
- **Text Chunking:** Splits extracted text into smaller chunks.
- **Embeddings:** Converts text chunks into numerical vector representations.
- **Vector Database:** Stores and searches embeddings using FAISS.
- **Semantic Search:** Retrieves relevant document chunks based on user questions.
- **AI Answer Generation:** Uses Groq LLM to generate context-based responses.
- **Source Information:** Uses document metadata such as page numbers and source names.
- **Interactive Chatbot:** Provides a user-friendly interface using Streamlit.
- **Cloud Deployment:** Deployed using Streamlit Community Cloud.

## 🏗️ Architecture

```text
             User
              |
              v
       Streamlit UI
              |
              v
        User Question
              |
              v
       Query Embedding
              |
              v
       FAISS Similarity Search
              |
              v
      Relevant Document Chunks
              |
              v
       Groq LLM Integration
              |
              v
        AI Generated Answer
              |
              v
             User
```

### Document Processing Pipeline

```text
        PDF Document
              |
              v
        Text Extraction
              |
              v
         Text Cleaning
              |
              v
        Text Chunking
              |
              v
       Embedding Generation
              |
              v
       FAISS Vector Index
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Groq API | LLM answer generation |
| NumPy | Numerical operations |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Document chunking |
| Git & GitHub | Version control |
| Streamlit Community Cloud | Deployment |

## 📂 Project Structure

```text
RAG practice/
│
├── app.py
├── main.py
├── build_database.py
├── document_processor.py
├── Embedding.py
├── vector_database.py
├── retriver.py
├── storage.py
├── generator.py
├── evaluation.py
│
├── data/
│   └── sample.pdf
│
├── database/
│   ├── faiss.index
│   └── chunks.json
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: `sample.pdf` represents the PDF document used by the project. Your actual local filename may differ. Keep private documents and secrets out of the public repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/gobinath0105-lgtm/RAG-1.git
```

### 2. Navigate to the project folder

```bash
cd RAG-1
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Groq API key

Create a `.env` file in the project root directory.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

**Security:** Never upload your `.env` file or expose your API key on GitHub.

### 7. Build the vector database

```bash
python build_database.py
```

This processes the configured PDF document, generates embeddings and builds the FAISS index.

### 8. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the chatbot.

## 🧠 How RAG Works in This Project

**1. Document Loading**

The PDF document is loaded and its text is extracted.

**2. Chunking**

The extracted text is divided into smaller chunks to make retrieval more effective.

**3. Embedding Generation**

The Sentence Transformer model `all-MiniLM-L6-v2` converts text chunks into 384-dimensional vectors.

**4. Vector Storage**

The embeddings are normalized and stored in a FAISS `IndexFlatIP` index.

**5. Retrieval**

The user's question is converted into an embedding. FAISS searches for the most similar document chunks using vector similarity.

**6. Answer Generation**

The retrieved chunks and user's question are passed to the Groq LLM. The LLM generates an answer using the provided context.

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key used to access Groq |

For Streamlit Community Cloud, configure the key under:

**App Settings → Secrets**

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## 🚀 Deployment

This application is deployed using Streamlit Community Cloud.

**Live Application:**

https://dojsepkvptduhemt2u6myf.streamlit.app/

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Retrieval-Augmented Generation.
- PDF processing and text extraction.
- Text chunking and metadata handling.
- Generating embeddings using Sentence Transformers.
- Building vector indexes using FAISS.
- Implementing semantic similarity search.
- Connecting an LLM through an API.
- Developing AI applications with Streamlit.
- Managing code using Git and GitHub.
- Deploying Python applications to the cloud.

## 🔮 Future Improvements

- Support for multiple PDF documents.
- Improved source citations and references.
- Better error handling and validation.
- More comprehensive RAG evaluation.
- FastAPI backend integration.
- React-based frontend.
- Database and user authentication.
- Docker-based deployment.

## 👨‍💻 Author

**Gobinath**

B.Tech – Artificial Intelligence and Data Science

- GitHub: https://github.com/gobinath0105-lgtm
- LinkedIn: https://linkedin.com/in/gobinath-gobi

---

⭐ If you find this project useful, consider giving the repository a star!
