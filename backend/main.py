import os
import tempfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Environment safety (Render / Cloud)
# --------------------------------------------------
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --------------------------------------------------
# LangChain imports (STABLE)
# --------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
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
# Embeddings (lazy + cloud-safe)
# --------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/tmp"
    )

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def detect_section(text: str) -> str:
    t = text.lower()[:300]
    if "abstract" in t:
        return "Abstract"
    if "method" in t or "methodology" in t:
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


def build_context(docs, max_chars=6000):
    """
    Explicitly constructs grounded context.
    Prevents hallucinations.
    """
    context_parts = []
    total_chars = 0

    for doc in docs:
        text = doc.page_content.strip()
        header = f"[{doc.metadata.get('paper')} | {doc.metadata.get('section')}]"
        block = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n".join(context_parts)

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
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    # 3. Vector store
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 4. Retrieve relevant chunks (CORRECT API)
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return {
            "answer": "Not found in the papers.",
            "citations": [],
            "confidence": 0.0
        }

    # 5. Build grounded context
    context_text = build_context(retrieved_docs)

    # 6. Prompt (STRICT grounding)
    prompt = PromptTemplate.from_template(
        """
You are an academic research assistant.

Answer ONLY using the context below.
If the answer is not explicitly stated, reply exactly:
"Not found in the papers."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 7. LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # 8. Final answer generation
    response = llm.invoke(
        prompt.format(
            context=context_text,
            question=query
        )
    )

    # 9. Citations
    citations = sorted(
        {
            f"{d.metadata.get('paper')} | {d.metadata.get('section')}"
            for d in retrieved_docs
        }
    )

    return {
        "answer": response.content.strip(),
        "citations": citations,
        "confidence": min(0.95, len(retrieved_docs) * 0.15)
    }
