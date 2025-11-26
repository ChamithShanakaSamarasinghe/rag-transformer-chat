import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Loading the PDF given to me for this assignment
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

# Now I am spliting the text into chunks
def split_into_chunks(documents, chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Testing the functions
    pdf_path = "data/transformer.pdf"
    
    docs = load_pdf(pdf_path)
    print(f"Loaded {len(docs)} pages from the PDF.")

    chunks = split_into_chunks(docs)
    print(f"Created {len(chunks)} text chunks.")

# No I am creating the embeddings for the assignment
def create_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Building the FAISS Vector store
def build_vector_store(chunks, embeddings, save_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

if __name__ == "__main__":
    #Using the PDF again for loading and chunking
    pdf_path = pdf_path = "C:/Users/raven/OneDrive/Desktop/rag-transformer-chat/data/transformer.pdf"
    docs = load_pdf(pdf_path)
    chunks = split_into_chunks(docs)

    #creating the embeddings
    embeddings = create_embeddings()
    print("Embeddings model loaded.")

    # Now I am writing the code to build the vector store
    vectorstore = build_vector_store(chunks, embeddings)
    print("Vector store created and saved locally.")