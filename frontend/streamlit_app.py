import streamlit as st
import requests
import uuid

# ---------------------------
# Config
# ---------------------------
API_BASE_URL = "http://127.0.0.1:8000"

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

st.set_page_config(
    page_title="Enterprise Document Q&A",
    page_icon="📄",
    layout="centered"
)

# ---------------------------
# Title
# ---------------------------
st.title("📄 Enterprise Document Q&A")
st.caption("RAG-powered document question answering using FAISS + Gemma")

st.divider()

# ---------------------------
# Upload Documents
# ---------------------------
st.header("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Upload & Index Documents"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Uploading and indexing documents..."):
            files = [
                ("files", (file.name, file.getvalue(), "application/pdf"))
                for file in uploaded_files
            ]

            response = requests.post(
                f"{API_BASE_URL}/upload-docs",
                files=files
            )

        if response.status_code == 200:
            st.success("Documents uploaded and indexed successfully!")
        else:
            st.error(f"Upload failed: {response.text}")

st.divider()

# ---------------------------
# Ask Question
# ---------------------------
st.header("❓ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What is the leave policy?"
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            response = requests.post(
                f"{API_BASE_URL}/ask-question",
                json={
                    "question": question,
                    "conversation_id": st.session_state.conversation_id
                }
            )

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.subheader("✅ Answer")
            st.write(answer)
        else:
            st.error(f"Error: {response.text}")
