# Homeopathy Knowledgebase - RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for homeopathy knowledge management, featuring document ingestion, semantic search, and AI-powered query answering.

## Features

- **Document Ingestion**: PDF text extraction and preprocessing
- **Semantic Indexing**: FAISS-based vector storage with embeddings
- **RAG Pipeline**: LLM integration for context-aware responses
- **FastAPI Backend**: RESTful API endpoints for all operations
- **Mobile App**: React Native app for mobile interaction
- **Scalable Architecture**: Modular design for easy model/vector store replacement

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r req.txt
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the Backend**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start the Mobile App**
   ```bash
   cd mobile_app
   npm install
   npm start
   ```

## API Endpoints

- `POST /api/ingest` - Upload and process PDF documents
- `GET /api/search` - Semantic search across documents
- `POST /api/query` - RAG-powered question answering
- `GET /api/documents` - List all ingested documents

## Architecture

- **Core**: LLM runner, FAISS ingestion, document processing
- **RAG Pipeline**: Connector, chunking, context injection
- **API Layer**: FastAPI with validation and error handling
- **Mobile Interface**: React Native app with authentication
- **Deployment**: VM-ready with monitoring and scaling

## Configuration

The system supports multiple LLM backends:
- OpenAI API
- Local models via Ollama (http://localhost:11434)
- Self-hosted models

## License

MIT License
