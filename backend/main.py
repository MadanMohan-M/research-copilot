import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# LangChain (safe imports)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq

# --------------------------------------------------
# App Setup
# --------------------------------------------------
app = FastAPI(title="Research Copilot (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Global Embeddings (LOAD ONCE – IMPORTANT)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

        for p in pages:
            p.metadata["paper"] = file.filename
            p.metadata["section"] = detect_section(p.page_content)
            documents.append(p)

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

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # 3. Prepare texts & metadata (IMPORTANT FIX)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # 4. FAISS Vector Store
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 5. Prompt (anti-hallucination)
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

    # 6. LLM (Groq)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 7. RAG Chain (Runnable API)
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    response = rag_chain.invoke(query)

    # 8. Retrieve source documents
    source_docs = retriever.invoke(query)

    citations = list(
        {
            f"{d.metadata.get('paper')} | {d.metadata.get('section')}"
            for d in source_docs
        }
    )

    return {
        "answer": response.content,
        "citations": citations,
        "confidence": min(0.95, len(source_docs) * 0.15)
    }
