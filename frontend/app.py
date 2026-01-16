import streamlit as st
import requests
import time
from uuid import uuid4

# =========================================================
# CONFIG
# =========================================================
BACKEND_URL = "http://localhost:8000/analyze"

st.set_page_config(
    page_title="Research Co-Pilot",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (CLEAN, HUMAN, ENTERPRISE STYLE)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Chat containers */
.stChatMessage {
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
}

/* User message */
.stChatMessage.user {
    background-color: #f1f5f9;
}

/* Assistant message */
.stChatMessage.assistant {
    background-color: #ffffff;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 500;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background-color: #f3f4f6;
}

/* Sticky input */
.stChatInputContainer {
    position: sticky;
    bottom: 0;
    background: white;
    padding-top: 12px;
    z-index: 100;
    border-top: 1px solid #e6e6e6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid #e6e6e6;
}

/* Expander */
details {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# =========================================================
# SIDEBAR (CONTROL CENTER)
# =========================================================
with st.sidebar:
    st.markdown("## Research Co-Pilot")
    st.caption("Evidence-based research assistant")

    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        for f in uploaded_files:
            st.markdown(f"- {f.name}")

    st.markdown("---")

    st.markdown("### Settings")
    top_k = st.slider("Context depth", 3, 10, 6)
    temperature = st.slider("Response variability", 0.0, 1.0, 0.0)

    st.markdown("---")
    if st.button("Start new session"):
        st.session_state.messages = []
        st.experimental_rerun()

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(
    """
    <h1 style="text-align:center;">Research Co-Pilot</h1>
    <p style="text-align:center;color:#6b7280;">
    Upload academic papers, ask research questions, and receive cited answers.
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
query = st.chat_input("Ask a research question")

# =========================================================
# HANDLE USER INPUT
# =========================================================
if query:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Validation
    if not st.session_state.uploaded_files:
        with st.chat_message("assistant"):
            st.warning("Please upload at least one PDF document.")
        st.stop()

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents and generating response"):
            time.sleep(0.4)

            files = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in st.session_state.uploaded_files
            ]

            data = {"query": query}

            try:
                response = requests.post(
                    BACKEND_URL,
                    files=files,
                    data=data,
                    timeout=300
                )
            except requests.exceptions.RequestException:
                st.error("Unable to connect to backend service.")
                st.stop()

        if response.status_code != 200:
            st.error("An error occurred while processing the request.")
            st.stop()

        result = response.json()

        # Typing effect (subtle)
        output = st.empty()
        text = ""
        for word in result["answer"].split():
            text += word + " "
            output.markdown(text)
            time.sleep(0.015)

        # Citations
        if result["citations"]:
            with st.expander("Sources"):
                for c in result["citations"]:
                    st.markdown(f"- {c}")

        # Follow-up prompts
        st.markdown("Suggested follow-up questions")
        cols = st.columns(3)
        followups = [
            "Summarize the key differences",
            "Which approach is more effective?",
            "What limitations are discussed?"
        ]
        for col, q in zip(cols, followups):
            if col.button(q):
                st.session_state.messages.append(
                    {"role": "user", "content": q}
                )
                st.experimental_rerun()

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
