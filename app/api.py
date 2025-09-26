import os
import random
import shutil
import asyncio
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from app.models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    DocumentListResponse, IngestionResponse, StatsResponse, 
    LLMTestResponse, ErrorResponse, ChatSessionResponse, ChatHistoryResponse, ChatMessage
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
        # Process, passing session_id to include conversational history in context
        result = rag_pipeline.process_query(request.query, request.top_k, session_id=request.session_id)
        print('here is the response =====', result)
        # Persist chat messages if database available
        try:
            if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'save_chat_message'):
                if request.session_id:
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'user', request.query)
                    answer_text = str(result.get('answer', ''))
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'ai', answer_text)
        except Exception as persist_err:
            logger.warning(f"Failed to save chat history: {persist_err}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/chat/new", response_model=ChatSessionResponse, tags=["RAG"])
async def create_chat_session():
    """Create and return a new chat session id."""
    try:
        # Simple session id generator; replace with robust server-side tracking if needed
        import uuid
        session_id = uuid.uuid4().hex
        return ChatSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating chat session: {str(e)}")

@app.get("/api/chat/{session_id}", response_model=ChatHistoryResponse, tags=["RAG"])
async def get_chat_history(session_id: str):
    """Return stored chat history for a given session."""
    try:
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_chat_history'):
            rows = rag_pipeline.vector_store.get_chat_history(session_id)
            messages = [ChatMessage(id=r.get('id'), session_id=session_id, role=r.get('role'), message=r.get('message')) for r in rows]
            return ChatHistoryResponse(session_id=session_id, messages=messages)
        else:
            return ChatHistoryResponse(session_id=session_id, messages=[])
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

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

def scrape_url(url: str, depth: int = 1, visited=None):
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)
    results = []
    
    # Create robust session
    session = requests.Session()
    
    # Add realistic headers to avoid blocking
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
    })
    
    try:
        # Add delay to avoid being blocked
        time.sleep(random.uniform(1, 3))
        
        res = session.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Remove unwanted elements
        for unwanted in soup(["script", "style", "nav", "header", "footer", "aside", 
                            "advertisement", "ads", "social-share", "cookie-banner", 
                            "noscript", "iframe", "form"]):
            unwanted.decompose()
        
        # Extract headings
        headings = []
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            heading_text = h.get_text(strip=True)
            if heading_text and len(heading_text) > 3:
                headings.append(heading_text)
        
        # Extract paragraphs with better filtering
        paragraphs = []
        for p in soup.find_all("p"):
            para_text = p.get_text(strip=True)
            if para_text and len(para_text) > 20:  # Filter out very short paragraphs
                paragraphs.append(para_text)
        
        # Also extract content from main, article, section tags
        main_content = []
        for tag in soup.find_all(["main", "article", "section", "div"]):
            if tag.get("class"):
                class_names = " ".join(tag.get("class", []))
                if any(keyword in class_names.lower() for keyword in ["content", "article", "main", "body", "text", "post", "entry"]):
                    content_text = tag.get_text(strip=True)
                    if content_text and len(content_text) > 50:
                        main_content.append(content_text)
        
        # Extract from lists that might contain useful information
        list_content = []
        for ul in soup.find_all(["ul", "ol"]):
            list_text = ul.get_text(strip=True)
            if list_text and len(list_text) > 30:
                list_content.append(list_text)
        
        # Combine all text
        all_text = " ".join(paragraphs + main_content + list_content)
        
        # Clean the text
        import re
        all_text = re.sub(r'\s+', ' ', all_text)  # Remove excessive whitespace
        all_text = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service|Subscribe|Newsletter', '', all_text, flags=re.IGNORECASE)
        
        # Remove Wikipedia navigation artifacts
        all_text = re.sub(r'From Wikipedia, the free encyclopedia|Jump to navigation|Jump to search', '', all_text, flags=re.IGNORECASE)
        
        # Only add if we have substantial content
        if all_text.strip() and len(all_text.strip()) > 100:
            results.append({
                "url": url,
                "headings": headings,
                "text": all_text.strip()
            })
            logger.info(f"Successfully scraped {url}: {len(all_text)} characters, {len(headings)} headings")
        else:
            logger.warning(f"Insufficient content scraped from {url}: only {len(all_text.strip())} characters")
        
        # Depth handling - follow links (only for same domain)
        if depth > 1:
            base_domain = urlparse(url).netloc
            links = soup.find_all("a", href=True)
            followed_links = 0
            
            for a in links:
                if followed_links >= 5:  # Limit to first 5 links to avoid infinite recursion
                    break
                    
                child_url = urljoin(url, a["href"])
                
                # Only follow links from same domain and avoid certain file types
                if (child_url.startswith("http") and 
                    urlparse(child_url).netloc == base_domain and
                    not any(child_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.xml', '.json'])):
                    
                    try:
                        child_results = scrape_url(child_url, depth-1, visited)
                        results.extend(child_results)
                        followed_links += 1
                    except Exception as e:
                        logger.warning(f"Failed to scrape child URL {child_url}: {e}")
                        continue
                    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        results.append({"url": url, "error": f"Request failed: {str(e)}"})
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        results.append({"url": url, "error": str(e)})
    finally:
        session.close()
    
    return results

@app.post("/api/weblink", tags=["Weblink"])
async def weblink_endpoint(request: Dict[str, Any]):
    """
    Schedule weblink scraping and ingestion as a background Celery task.
    Returns a job_id to poll with any Celery-aware progress endpoint.
    """
    try:
        url = request.get("query")
        depth = int(request.get("depth", 1))
        max_workers = int(request.get("max_workers", 4))
        batch_size = int(request.get("batch_size", 100))

        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")

        task = celery_app.send_task(
            "app.tasks.weblink_ingestion_task",
            args=[url, depth, max_workers, batch_size]
        )
        return {"job_id": task.id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling weblink async ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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