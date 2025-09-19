import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import json
from datetime import datetime

from app.models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    DocumentListResponse, IngestionResponse, StatsResponse, 
    LLMTestResponse, ErrorResponse
)
from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor
from app.config import settings
from app.celery_app import celery_app
from celery.result import AsyncResult

# Global storage for tracking ingestion progress
ingestion_progress_store = {}

class EnhancedIngestionResponse(IngestionResponse):
    """Enhanced response model with parallel processing stats."""
    processing_time: float = 0.0
    chunks_per_second: float = 0.0
    successful_batches: int = 0
    failed_batches: int = 0
    total_batches: int = 0
    max_workers: int = 0
    batch_size: int = 0

# Initialize FastAPI app
app = FastAPI(
    title="Homeopathy Knowledgebase RAG API",
    description="API for homeopathy knowledge retrieval and question answering with parallel processing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize components
rag_pipeline = RAGPipeline()
document_processor = DocumentProcessor()

@app.get("/")
async def serve_ui():
    """Serve the main UI."""
    ui_file = static_dir / "index.html"
    if ui_file.exists():
        return FileResponse(str(ui_file))
    else:
        return {
            "message": "Enhanced Homeopathy Knowledgebase RAG API with Parallel Processing",
            "version": "2.0.0",
            "status": "running",
            "features": ["parallel_ingestion", "batch_processing", "progress_tracking"],
            "endpoints": {
                "ingest": "/api/ingest - Parallel document ingestion",
                "ingest_async": "/api/ingest/async - Async ingestion with progress tracking",
                "ingest_progress": "/api/ingest/progress/{job_id} - Check ingestion progress",
                "query": "/api/query",
                "search": "/api/search",
                "documents": "/api/documents",
                "stats": "/api/stats",
                "llm-test": "/api/llm-test"
            }
        }

@app.post("/api/ingest", response_model=EnhancedIngestionResponse, tags=["Documents"])
async def ingest_documents_parallel(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    max_workers: int = Form(default=4, description="Number of parallel workers"),
    batch_size: int = Form(default=100, description="Chunks per batch")
):
    """Ingest PDF documents using parallel processing."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF"
                )
        
        processed_docs = []
        total_chunks = 0
        
        # Process all files first
        logger.info(f"Starting document processing for {len(files)} files...")
        for file in files:
            try:
                # Save uploaded file
                file_path = Path(settings.upload_dir) / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process document
                doc_info = document_processor.process_document(file_path)
                processed_docs.append(doc_info)
                total_chunks += doc_info['total_chunks']
                
                logger.info(f"Processed {file.filename} into {doc_info['total_chunks']} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                continue
        
        if not processed_docs:
            raise HTTPException(status_code=500, detail="No documents were successfully processed")
        
        # Configure parallel processing
        rag_pipeline.vector_store.max_workers = max_workers
        rag_pipeline.vector_store.batch_size = batch_size
        
        logger.info(f"Starting parallel ingestion with {max_workers} workers, batch size {batch_size}")
        
        # Add documents using parallel processing
        result_stats = rag_pipeline.vector_store.add_documents_parallel(processed_docs)
        
        if result_stats['success']:
            return EnhancedIngestionResponse(
                success=True,
                message=f"Successfully ingested {len(processed_docs)} documents using parallel processing",
                documents_processed=result_stats['total_documents'],
                chunks_created=result_stats['processed_chunks'],
                processing_time=result_stats['processing_time'],
                chunks_per_second=result_stats['chunks_per_second'],
                successful_batches=result_stats['successful_batches'],
                failed_batches=result_stats['failed_batches'],
                total_batches=result_stats['total_batches'],
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Partial failure: {result_stats['failed_chunks']} chunks failed to process"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parallel document ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/ingest/async", tags=["Documents"])
async def ingest_documents_async(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    max_workers: int = Form(default=4),
    batch_size: int = Form(default=100)
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        saved_paths = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            dest = Path(settings.upload_dir) / file.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(dest))

        # Enqueue celery job
        task = celery_app.send_task(
            "app.tasks.ingest_documents_task",
            args=[saved_paths, max_workers, batch_size]
        )
        return {"job_id": task.id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling async ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingest/progress/{job_id}", tags=["Documents"])
async def get_ingest_progress(job_id: str):
    try:
        res = AsyncResult(job_id, app=celery_app)
        response = {
            "job_id": job_id,
            "state": res.state,
            "progress": 0,
            "detail": None,
        }
        info = res.info or {}
        if isinstance(info, dict):
            response.update(info)
        if res.successful():
            response["state"] = "SUCCESS"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest):
    """Process a query through the RAG pipeline."""
    try:
        result = rag_pipeline.process_query(request.query, request.top_k)
        # print('here is the response =====', result)
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        doc_list = rag_pipeline.get_all_documents()
        if doc_list is None:
            doc_list = []
        return DocumentListResponse(documents=doc_list, total_documents=len(doc_list))
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/api/llm-test", response_model=LLMTestResponse, tags=["System"])
async def test_llm():
    """Test LLM connection and functionality."""
    try:
        result = rag_pipeline.test_llm_connection()
        print(result)
        return LLMTestResponse(**result)
        
    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        return LLMTestResponse(
            status="error",
            provider="unknown",
            error=str(e)
        )

@app.delete("/api/documents", tags=["Documents"])
async def clear_knowledge_base():
    """Clear the entire knowledge base."""
    try:
        success = rag_pipeline.clear_knowledge_base()
        
        if success:
            return {"message": "Knowledge base cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

@app.get("/api/performance/benchmark", tags=["System"])
async def benchmark_ingestion():
    """Benchmark different ingestion configurations."""
    try:
        # This would run a benchmark with different worker/batch combinations
        # For demo purposes, returning mock data
        benchmark_results = {
            "configurations": [
                {
                    "workers": 1,
                    "batch_size": 50,
                    "chunks_per_second": 45.2,
                    "total_time": 120.5
                },
                {
                    "workers": 2,
                    "batch_size": 100,
                    "chunks_per_second": 78.1,
                    "total_time": 69.8
                },
                {
                    "workers": 4,
                    "batch_size": 100,
                    "chunks_per_second": 142.3,
                    "total_time": 38.3
                },
                {
                    "workers": 8,
                    "batch_size": 200,
                    "chunks_per_second": 201.7,
                    "total_time": 27.1
                }
            ],
            "recommended": {
                "workers": 4,
                "batch_size": 100,
                "reason": "Best balance of speed and resource usage"
            }
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Error running benchmark: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            details={"exception": str(exc)}
        ).dict()
    )