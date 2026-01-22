from dotenv import load_dotenv
import os
import requests

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from app.prompt import ENTERPRISE_QA_PROMPT

# Load environment variables
load_dotenv()

FAISS_PATH = "embeddings/faiss_index"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def call_gemma(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Enterprise RAG Demo",
    }

    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]

def answer_question(question: str) -> str:
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = ENTERPRISE_QA_PROMPT.format(
        context=context,
        question=question
    )

    return call_gemma(final_prompt)

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = answer_question(question)
    print("\nAnswer:\n", answer)
