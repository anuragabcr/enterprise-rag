from dotenv import load_dotenv
import os
import requests

import hashlib
from app.cache import redis_client, get_conversation, save_conversation

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from app.prompt import ENTERPRISE_QA_PROMPT, CONVERSATIONAL_RAG_PROMPT

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

def format_history(messages: list, max_turns: int = 5) -> str:
    recent = messages[-(max_turns * 2):]
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in recent
    )


def call_gemma_langchain(prompt: str) -> str:
    llm = ChatOpenAI(
        model="google/gemma-3-27b-it:free",
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Enterprise RAG Demo",
        },
        openai_proxy=""
    )

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    return response.content

def get_cache_key(question: str) -> str:
    question_hash = hashlib.sha256(question.encode()).hexdigest()
    return f"rag_answer:{question_hash}"

def answer_question(question: str) -> str:
    cache_key = get_cache_key(question)

    cached_answer = redis_client.get(cache_key)
    if cached_answer:
        return cached_answer 
    
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = ENTERPRISE_QA_PROMPT.format(
        context=context,
        question=question
    )

    answer = call_gemma(final_prompt)
    # answer = call_gemma_langchain(final_prompt)

    redis_client.setex(
        cache_key,
        3600,
        answer
    )

    return answer

def answer_question_conv(question: str, conversation_id: str) -> str:
    history_messages = get_conversation(conversation_id)
    history_text = format_history(history_messages)

    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = CONVERSATIONAL_RAG_PROMPT.format(
        history=history_text,
        context=context,
        question=question
    )

    answer = call_gemma(final_prompt)

    history_messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])

    save_conversation(conversation_id, history_messages)

    return answer

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = answer_question(question)
    print("\nAnswer:\n", answer)
