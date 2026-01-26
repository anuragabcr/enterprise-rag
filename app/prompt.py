from langchain.prompts import PromptTemplate

# Base RAG prompt (default)
ENTERPRISE_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an enterprise AI assistant.

STRICT RULES:
- Use ONLY the information provided in the context
- Do NOT use external knowledge
- If the answer is not in the context, say:
  "Not found in provided documents"

Context:
{context}

Question:
{question}

Answer:
"""
)

CONVERSATIONAL_RAG_PROMPT = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are an enterprise AI assistant.

Conversation history:
{history}

Use ONLY the information from the context below to answer.
If the answer is not in the context, say:
"Not found in provided documents".

Context:
{context}

Current Question:
{question}

Answer:
"""
)
