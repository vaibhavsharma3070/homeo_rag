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

# Initialize components with parallel processing
rag_pipeline = RAGPipeline()  # Assuming this now uses ParallelPGVectorStore
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
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    max_workers: int = Form(default=4, description="Number of parallel workers"),
    batch_size: int = Form(default=100, description="Chunks per batch")
):
    """Start asynchronous document ingestion with progress tracking."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Generate unique job ID
        job_id = f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(files)}files"
        
        # Initialize progress tracking
        ingestion_progress_store[job_id] = {
            "status": "processing",
            "total_files": len(files),
            "processed_files": 0,
            "total_chunks": 0,
            "processed_chunks": 0,
            "failed_chunks": 0,
            "current_batch": 0,
            "total_batches": 0,
            "start_time": datetime.now().isoformat(),
            "estimated_completion": None,
            "error": None
        }
        
        # Save files and create processing task
        saved_files = []
        for file in files:
            file_path = Path(settings.upload_dir) / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append({
                "filename": file.filename,
                "path": str(file_path)
            })
        
        # Add background task
        background_tasks.add_task(
            process_documents_background, 
            job_id, 
            saved_files, 
            max_workers, 
            batch_size
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Async ingestion started for {len(files)} files",
            "progress_endpoint": f"/api/ingest/progress/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting async ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting async ingestion: {str(e)}")

@app.get("/api/ingest/progress/{job_id}", tags=["Documents"])
async def get_ingestion_progress(job_id: str):
    """Get progress of an async ingestion job."""
    if job_id not in ingestion_progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    progress = ingestion_progress_store[job_id]
    
    # Calculate additional metrics
    if progress["status"] == "processing" and progress["processed_chunks"] > 0:
        start_time = datetime.fromisoformat(progress["start_time"])
        elapsed = (datetime.now() - start_time).total_seconds()
        chunks_per_second = progress["processed_chunks"] / elapsed if elapsed > 0 else 0
        
        if chunks_per_second > 0 and progress["total_chunks"] > progress["processed_chunks"]:
            remaining_chunks = progress["total_chunks"] - progress["processed_chunks"]
            eta_seconds = remaining_chunks / chunks_per_second
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            progress["estimated_completion"] = eta.isoformat()
        
        progress["processing_speed"] = f"{chunks_per_second:.1f} chunks/sec"
        progress["elapsed_time"] = f"{elapsed:.1f}s"
    
    return progress

async def process_documents_background(
    job_id: str, 
    saved_files: List[Dict[str, str]], 
    max_workers: int, 
    batch_size: int
):
    """Background task for processing documents."""
    try:
        # Update progress callback
        def progress_callback(progress_info):
            if job_id in ingestion_progress_store:
                ingestion_progress_store[job_id].update({
                    "processed_chunks": progress_info["processed_chunks"],
                    "total_chunks": progress_info["total_chunks"],
                    "current_batch": progress_info["current_batch"],
                    "total_batches": progress_info["total_batches"]
                })
        
        # Process documents
        processed_docs = []
        total_chunks = 0
        
        for file_info in saved_files:
            try:
                doc_info = document_processor.process_document(Path(file_info["path"]))
                processed_docs.append(doc_info)
                total_chunks += doc_info['total_chunks']
                
                # Update progress
                ingestion_progress_store[job_id]["processed_files"] += 1
                ingestion_progress_store[job_id]["total_chunks"] = total_chunks
                
            except Exception as e:
                logger.error(f"Error processing {file_info['filename']}: {e}")
                continue
        
        if not processed_docs:
            raise Exception("No documents were successfully processed")
        
        # Configure and run parallel ingestion
        rag_pipeline.vector_store.max_workers = max_workers
        rag_pipeline.vector_store.batch_size = batch_size
        
        result_stats = rag_pipeline.vector_store.add_documents_parallel(
            processed_docs, 
            progress_callback=progress_callback
        )
        
        # Update final status
        ingestion_progress_store[job_id].update({
            "status": "completed" if result_stats['success'] else "completed_with_errors",
            "total_documents": result_stats['total_documents'],
            "processed_chunks": result_stats['processed_chunks'],
            "failed_chunks": result_stats['failed_chunks'],
            "processing_time": result_stats['processing_time'],
            "chunks_per_second": result_stats['chunks_per_second'],
            "successful_batches": result_stats['successful_batches'],
            "failed_batches": result_stats['failed_batches'],
            "completion_time": datetime.now().isoformat()
        })
        
    except Exception as e:
        # Update error status
        ingestion_progress_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.now().isoformat()
        })

@app.get("/api/ingest/progress/stream/{job_id}", tags=["Documents"])
async def stream_ingestion_progress(job_id: str):
    """Stream real-time progress updates for an ingestion job."""
    if job_id not in ingestion_progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def generate_progress_stream():
        while True:
            if job_id not in ingestion_progress_store:
                break
                
            progress = ingestion_progress_store[job_id]
            yield f"data: {json.dumps(progress)}\n\n"
            
            if progress["status"] in ["completed", "completed_with_errors", "failed"]:
                break
                
            await asyncio.sleep(1)  # Update every second
    
    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest):
    """Process a query through the RAG pipeline."""
    try:
        result = rag_pipeline.process_query(request.query, request.top_k)
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(request: SearchRequest):
    """Search documents without generating a response."""
    try:
        results = rag_pipeline.search_documents(request.query, request.top_k)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

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

@app.get("/api/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: int):
    """Get a specific document by ID."""
    try:
        document = rag_pipeline.get_document_by_id(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunks = rag_pipeline.get_document_chunks(doc_id)
        
        return {
            "document": document,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get enhanced knowledge base statistics including parallel processing info."""
    try:
        stats = rag_pipeline.get_knowledge_base_stats()
        
        if 'error' in stats:
            raise HTTPException(status_code=500, detail=stats['error'])
        
        # Add parallel processing stats if available
        if hasattr(rag_pipeline.vector_store, 'max_workers'):
            stats.update({
                'parallel_processing': {
                    'max_workers': rag_pipeline.vector_store.max_workers,
                    'batch_size': rag_pipeline.vector_store.batch_size,
                    'enabled': True
                }
            })
        
        return StatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/ingestion/jobs", tags=["Documents"])
async def list_ingestion_jobs():
    """List all ingestion jobs and their status."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job_info["status"],
                "total_files": job_info.get("total_files", 0),
                "total_chunks": job_info.get("total_chunks", 0),
                "processed_chunks": job_info.get("processed_chunks", 0),
                "start_time": job_info.get("start_time"),
                "completion_time": job_info.get("completion_time")
            }
            for job_id, job_info in ingestion_progress_store.items()
        ],
        "total_jobs": len(ingestion_progress_store)
    }

@app.delete("/api/ingestion/jobs/{job_id}", tags=["Documents"])
async def delete_ingestion_job(job_id: str):
    """Delete an ingestion job from tracking."""
    if job_id not in ingestion_progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = ingestion_progress_store[job_id]["status"]
    if job_status == "processing":
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    
    del ingestion_progress_store[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/api/llm-test", response_model=LLMTestResponse, tags=["System"])
async def test_llm():
    """Test LLM connection and functionality."""
    try:
        result = rag_pipeline.test_llm_connection()
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