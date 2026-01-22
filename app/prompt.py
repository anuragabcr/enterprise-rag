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
