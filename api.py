import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Trying new langchain-huggingface first
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Transformer API", version="1.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    context: str


@app.on_event("startup")
def load_resources():
    """Load embeddings, vectorstore and configure Gemini."""
    global embeddings, vectorstore, gen_model

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY is not set!")

    genai.configure(api_key=gemini_key)

    gen_model = os.environ.get("GENIE_MODEL_ID", "models/gemini-2.5-flash")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieving similar chunks
    docs = vectorstore.similarity_search(req.question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Building RAG prompt
    prompt = f"""
Using ONLY the context provided below, answer the question.
Do not hallucinate or add extra details.

Context:
{context}

Question: {req.question}

Answer:
"""

    model = genai.GenerativeModel(gen_model)
    response = model.generate_content(prompt)

    answer = response.text if hasattr(response, "text") else str(response)

    return AskResponse(
        question=req.question,
        answer=answer.strip(),
        context=context
    )
