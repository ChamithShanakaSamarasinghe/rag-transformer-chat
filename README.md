This project implements a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on uploaded documents.
It consists of:

    1. A FastAPI backend (Dockerized)

    2. A Streamlit UI chatbot (runs locally)

    3. A FAISS vector store

    4. Sentence-transformers for embeddings

    5. Gemini API for generation

Running the Backend with Docker (FastAPI)
    1. Build the docker image
        docker build -t rag-app .

    2. Run the container
        export GEMINI_API_KEY="YOUR_API_KEY"
        docker run -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" rag-app

    API will run at
        http://localhost:8000/docs


Running the chatbot (Streamlit UI)
    If using CMD, then
    go to the folder then type streamlit run app_streamlit.py 

    If using ubuntu, then
    go to the folder using "cd /mnt/c/rag-transformer-chat/rag-transformer-chat" 
    then activate the virtual enviornment using "source venv/bin/activate"
    then run the chatbot UI using "streamlit run app_streamlit.py"

    once running the localhost will be http://localhost:8501


How the System Works:

    1. Documents in the data/ folder are loaded.

    2. SentenceTransformers create embeddings.

    3. Embeddings are stored in a FAISS vector database.

    4. User question → converted to embedding → FAISS retrieves best chunks.

    5. Retrieved context + question → sent to Gemini model.

    6. Gemini returns the final answer.


Technologies & Libraries

    Backend

        1. FastAPI

        2. LangChain

        3. FAISS

        4. Sentence-transformers

        5. Google Gemini API


    Frontend

        1. Streamlit

    
    Deployment

        1. Docker

        2. python:3.11-slim base image


Example Queries to Test:

    1. “Summarize the document inside the data folder.”

    2. “Explain the main idea in simple terms.”

    3. “What are the key findings from the document?”


Done by Chamith Shanaka Samarasinghe
BEng (Hons) Software Engineering
2025