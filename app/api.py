from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from pydantic import BaseModel

from app.ingest import ingest_documents
from app.query import answer_question

DATA_DIR = "data/documents"

app = FastAPI(
    title="Enterprise RAG API",
    description="Document Q&A using FAISS + HuggingFace + OpenRouter (Gemma)",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/upload-docs")
def upload_docs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    ingest_documents()

    return {
        "message": "Documents uploaded and indexed successfully",
        "files": [file.filename for file in files]
    }

@app.post("/ask-question")
def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }

