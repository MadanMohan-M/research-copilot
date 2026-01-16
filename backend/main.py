import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Environment safety (CRITICAL for Render)
# --------------------------------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --------------------------------------------------
# LangChain imports (STABLE & SAFE)
# --------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Research Copilot (RAG API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Lazy Embeddings Loader (RENDER-SAFE)
# --------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp"  # Writable on Render
    )

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def detect_section(text: str) -> str:
    t = text.lower()[:300]
    if "abstract" in t:
        return "Abstract"
    if "method" in t:
        return "Methodology"
    if "result" in t:
        return "Results"
    if "conclusion" in t:
        return "Conclusion"
    return "General"


def load_pdfs(files: List[UploadFile]):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        pages = loader.load()

        for page in pages:
            page.metadata["paper"] = file.filename
            page.metadata["section"] = detect_section(page.page_content)
            documents.append(page)

    return documents

# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
@app.post("/analyze")
async def analyze_papers(
    files: List[UploadFile] = File(...),
    query: str = Form(...)
):
    # 1. Load PDFs
    documents = load_pdfs(files)

    if not documents:
        return {
            "answer": "No readable text found in the uploaded papers.",
            "citations": [],
            "confidence": 0.0
        }

    # 2. Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # 3. Prepare text + metadata
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # 4. Embeddings (lazy)
    embeddings = get_embeddings()

    # 5. FAISS Vector Store
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 6. Retrieve documents FIRST (IMPORTANT)
    source_docs = retriever.invoke(query)

    if not source_docs:
        return {
            "answer": "Not found in the uploaded papers.",
            "citations": [],
            "confidence": 0.0
        }

    # 7. Prompt (anti-hallucination)
    prompt = PromptTemplate.from_template(
        """
You are an academic research assistant.

Answer ONLY from the given context.
Do NOT use external knowledge.
If the answer is not found, say:
"Not found in the papers".

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 8. LLM (Groq)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 9. RAG Chain (Runnable API)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    # 10. Safe execution
    try:
        response = rag_chain.invoke(query)
        answer = response.content
    except Exception:
        answer = "Not found in the uploaded papers."

    # 11. Citations
    citations = list(
        {
            f"{d.metadata.get('paper')} | {d.metadata.get('section')}"
            for d in source_docs
        }
    )

    return {
        "answer": answer,
        "citations": citations,
        "confidence": min(0.95, len(source_docs) * 0.15)
    }
