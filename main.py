import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Loading the FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Configuring the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# I am trying to add the Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# RAG query function
def rag_query(question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant using information from a research paper.

Context:
{context}

Question: {question}

Answer concisely and accurately using ONLY the context provided.
"""

    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    question = input("Ask a question about the Transformer paper: ")
    answer = rag_query(question)
    print("\nAnswer:")
    print(answer)
