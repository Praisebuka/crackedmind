FROM python:3.11-slim

WORKDIR /app

# System deps (needed for faiss-cpu + sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy app
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data dir for FAISS index + rating model
RUN mkdir -p data

EXPOSE 8000

# Non-root user (security hardening)
RUN useradd -m -u 1000 crackedmind && chown -R crackedmind:crackedmind /app
USER crackedmind

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
