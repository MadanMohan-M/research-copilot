import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Render safety
# -----------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# -----------------------------
# LangChain imports
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Research Copilot (RAG API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Embeddings (lazy, Render-safe)
# -----------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp"
    )

# -----------------------------
# Helpers
# -----------------------------
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
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        pages = loader.load()

        for p in pages:
            p.metadata["paper"] = file.filename
            p.metadata["section"] = detect_section(p.page_content)
            docs.append(p)

    return docs


def build_context(docs, max_chars=5000):
    context = ""
    for d in docs:
        if len(context) > max_chars:
            break
        context += f"\n[{d.metadata['paper']} | {d.metadata['section']}]\n"
        context += d.page_content.strip()
    return context.strip()

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/analyze")
async def analyze_papers(
    files: List[UploadFile] = File(...),
    query: str = Form(...)
):
    documents = load_pdfs(files)
    if not documents:
        return {
            "answer": "No readable content found.",
            "citations": [],
            "confidence": 0.0
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ✅ CORRECT retrieval (NO deprecated APIs)
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {
            "answer": "Not found in the papers.",
            "citations": [],
            "confidence": 0.0
        }

    context = build_context(retrieved_docs)

    prompt = PromptTemplate.from_template(
        """
Answer ONLY using the context below.
If the answer is not present, reply exactly:
"Not found in the papers."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    citations = list({
        f"{d.metadata['paper']} | {d.metadata['section']}"
        for d in retrieved_docs
    })

    return {
        "answer": response.content,
        "citations": citations,
        "confidence": min(0.95, len(retrieved_docs) * 0.15)
    }
