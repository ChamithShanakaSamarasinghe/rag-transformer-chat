FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install CPU-only Torch + SentenceTransformers + all other deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir sentence-transformers && \
    pip install --no-cache-dir -r requirements.txt

# Copy remaining project files
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
