# Research Co-Pilot

Research Co-Pilot is a production-ready AI system for analyzing academic papers using Retrieval-Augmented Generation (RAG).  
It allows users to upload research papers (PDFs), ask deep questions, and receive evidence-backed answers with citations.

This project is designed as a **research-grade AI tool**, not a demo.

---

## Features

- RAG-based question answering using LangChain
- Upload and analyze multiple academic PDFs
- Evidence-grounded answers with source citations
- Chat-style conversational interface
- Follow-up question suggestions
- Session-based chat history
- Clean, enterprise-style Streamlit UI
- FastAPI backend with FAISS vector search
- Groq-hosted LLM for fast inference

---

## Tech Stack

### Backend
- Python
- FastAPI
- LangChain
- FAISS (CPU)
- HuggingFace Sentence Transformers
- Groq LLM API

### Frontend
- Streamlit
- Custom CSS (SaaS-style UI)
- REST API integration

### Deployment
- GitHub
- Render (Backend + Frontend as separate services)

---

## Architecture

User
└── Streamlit Frontend
└── FastAPI Backend
├── PDF Loader
├── Text Chunking
├── FAISS Vector Store
├── Retriever
└── LLM (Groq)

yaml
Copy code

---

## How It Works

1. User uploads PDF research papers
2. PDFs are split into semantic chunks
3. Chunks are embedded using sentence-transformers
4. FAISS stores vectors for similarity search
5. Relevant chunks are retrieved for a query
6. LLM generates answers using retrieved context only
7. Source citations are returned alongside answers

---

## Local Setup

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/research-copilot.git
cd research-copilot
Backend Setup
bash
Copy code
cd backend
pip install -r requirements.txt
set GROQ_API_KEY=your_api_key_here
uvicorn main:app --reload
Backend runs at:

arduino
Copy code
http://localhost:8000
Frontend Setup
bash
Copy code
cd frontend
pip install -r requirements.txt
streamlit run app.py
Frontend runs at:

arduino
Copy code
http://localhost:8501
Deployment on Render
Backend and frontend are deployed as separate Render Web Services

Environment variable GROQ_API_KEY is set in Render

Backend exposed via /analyze API

Frontend connects to backend using deployed URL

Limitations
Free Render instances may experience cold starts

Large PDFs may increase response latency

No persistent database (sessions stored in memory)

Future Improvements
Persistent chat history with database

PDF page-level citations

User authentication

Dark/light mode toggle

Streaming token-level responses

Support for additional document formats

Author
Built by Mokka Madan Yadav
AI / ML Developer

License
This project is licensed under the MIT License.
