import streamlit as st
import google.generativeai as genai
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#  Configuring the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-2.0-flash"

model = genai.GenerativeModel(MODEL_NAME)


#  LOAD EMBEDDINGS + VECTORSTORE
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return embeddings, vectorstore


embeddings, vectorstore = load_system()

#  RAG QUERY FUNCTION
def rag_query(question):
    similar_docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in similar_docs)

    prompt = f"""
    You are an AI assistant helping with understanding the Transformer research paper.

    Use the following context to answer the question:

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Provide a clear and simple explanation.
    """

    response = model.generate_content(prompt)
    return response.text


#  STREAMLIT UI
st.set_page_config(page_title="Transformer RAG Chatbot", layout="wide")

st.title("📘 Transformer Paper RAG Chatbot")
st.write("Ask any question about the Transformer architecture, and I will retrieve and explain it using the original research paper.")


#  CHAT INPUT
user_question = st.text_input("🔍 Enter your question about the Transformer paper:")

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = rag_query(user_question)

        st.subheader("📌 Answer")
        st.write(answer)
