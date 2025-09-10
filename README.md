# Homeo RAG Setup

## Prerequisites
- Git, Docker, Python 3.8+, psql

## Quick Setup

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
ollama run phi3:mini
```

### 2. Create .env file
```env
# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini

# Vector Store Configuration
VECTOR_BACKEND=pgvector
FAISS_INDEX_PATH=./data/faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API Configuration
API_HOST=0.0.0.0
API_PORT=8090
DEBUG=true

# Storage Configuration
UPLOAD_DIR=./data/uploads
PROCESSED_DIR=./data/processed
LOG_LEVEL=INFO

# Database (pgvector) Configuration
DATABASE_URL=postgresql+psycopg2://username:password@localhost:5432/dbname
POSTGRES_USER=username
POSTGRES_PASSWORD=password
POSTGRES_DB=dbname

# Security Configuration
SECRET_KEY=supersecretkey123
ACCESS_TOKEN_EXPIRE_MINUTES=30

```

### 3. Docker Setup
```bash
# Make sure you have docker-compose.yml file
docker compose up -d
sleep 30
psql -h localhost -p 5432 -U dev_user -d embedding_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 4. Python Setup
```bash
git clone https://github.com/vaibhavsharma3070/homeo_rag.git
cd homeo_rag
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 5. Run Application
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

## Access Points
- API: http://localhost:8090
- Docs: http://localhost:8090/docs
- Ollama: http://localhost:11434

## One-Line Setup
```bash
curl -fsSL https://ollama.com/install.sh | sh && ollama pull phi3:mini && git clone https://github.com/vaibhavsharma3070/homeo_rag.git && cd homeo_rag && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && docker compose up -d && sleep 30 && psql -h localhost -p 5432 -U dev_user -d embedding_db -c "CREATE EXTENSION IF NOT EXISTS vector;" && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```