import os
import random
import shutil
import asyncio
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request, Header
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
import json
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from jose import jwt

from app.models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    DocumentListResponse, IngestionResponse, StatsResponse, 
    LLMTestResponse, ErrorResponse, ChatSessionResponse, ChatHistoryResponse, ChatMessage,
    ChatSessionsListResponse, ChatSessionInfo, LoginRequest, LoginResponse, RegisterRequest, UserInfo
)
from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor
from app.config import settings
from app.celery_app import celery_app
from celery.result import AsyncResult

# Global storage for tracking ingestion progress
ingestion_progress_store = {}

# Track corrupted tasks to avoid log spam
corrupted_tasks_cache = set()

# Worker readiness tracking
_worker_ready = False

# JWT Security
security = HTTPBearer(auto_error=False)

def create_access_token(username: str, user_id: int) -> str:
    """Create JWT access token."""
    expire = datetime.utcnow() + timedelta(days=30)  # 30 days expiry
    payload = {
        "sub": username,
        "user_id": user_id,
        "exp": expire
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return user info."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from token."""
    try:
        if not credentials:
            logger.debug("get_current_user: No credentials provided")
            return None
        
        token = credentials.credentials
        if not token:
            logger.debug("get_current_user: No token in credentials")
            return None
            
        payload = verify_token(token)
        if not payload:
            logger.debug("get_current_user: Token verification failed")
            return None
        
        username = payload.get("sub")
        if not username:
            logger.debug("get_current_user: No username in token payload")
            return None
        
        # Get user info from database
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_user_by_username'):
            user = rag_pipeline.vector_store.get_user_by_username(username)
            if user:
                logger.debug(f"get_current_user: Found user {username}, role: {user.get('role')}")
            else:
                logger.warning(f"get_current_user: User {username} not found in database")
            return user
        logger.warning("get_current_user: vector_store or get_user_by_username not available")
        return None
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}", exc_info=True)
        return None

def check_worker_ready():
    """Check if Celery worker is ready"""
    global _worker_ready
    if _worker_ready:
        return True
    
    try:
        # Send a simple ping task to check worker availability
        inspect = celery_app.control.inspect()
        active = inspect.active()
        if active and len(active) > 0:
            _worker_ready = True
            logger.info("Celery worker is ready")
            return True
    except Exception as e:
        logger.warning(f"Worker not ready yet: {e}")
    
    return False

def cleanup_corrupted_task(job_id: str):
    """Attempt to clean up a corrupted task result from Redis."""
    try:
        from celery.result import AsyncResult
        res = AsyncResult(job_id, app=celery_app)
        # Try to forget/delete the corrupted result
        try:
            res.forget()
            logger.info(f"Cleaned up corrupted task result for job {job_id}")
        except Exception as forget_error:
            logger.debug(f"Could not forget task {job_id}: {forget_error}")
            # Try to delete from Redis directly if available
            try:
                backend = celery_app.backend
                if hasattr(backend, 'delete'):
                    backend.delete(job_id)
                    logger.info(f"Deleted corrupted task result from backend for job {job_id}")
            except Exception as delete_error:
                logger.debug(f"Could not delete task {job_id} from backend: {delete_error}")
    except Exception as e:
        logger.debug(f"Error cleaning up corrupted task {job_id}: {e}")

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

# Middleware to handle large file uploads and improve error messages
class LargeFileUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            error_msg = str(e)
            # Check if it's a request body size error
            if "request body too large" in error_msg.lower() or "413" in error_msg:
                logger.error(f"Request body too large: {error_msg}")
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": "File upload too large. Please try uploading smaller files or contact the administrator to increase server limits."
                    }
                )
            # Check for connection/timeout errors
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                logger.error(f"Connection/timeout error: {error_msg}")
                return JSONResponse(
                    status_code=504,
                    content={
                        "detail": "Upload timeout. The file upload took too long. Please try again with smaller files or check your network connection."
                    }
                )
            raise

app.add_middleware(LargeFileUploadMiddleware)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize components
rag_pipeline = RAGPipeline()
document_processor = DocumentProcessor()

@app.get("/")
async def serve_login():
    """Serve the login page."""
    login_file = static_dir / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    else:
        raise HTTPException(status_code=404, detail="Login page not found")

@app.get("/login.html")
async def serve_login_alias():
    """Serve the login page (alias for /)."""
    login_file = static_dir / "login.html"
    if login_file.exists():
        return FileResponse(str(login_file))
    else:
        raise HTTPException(status_code=404, detail="Login page not found")

@app.get("/app")
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
                "llm-test": "/api/llm-test",
                "scrape-test": "/api/scrape-test - Test web scraping functionality",
                "weblink": "/api/weblink - Scrape and ingest web content"
            }
        }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint that returns true."""
    return {"status": True}

@app.post("/api/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """Login endpoint to authenticate user with email and password."""
    try:
        if not hasattr(rag_pipeline, 'vector_store') or not hasattr(rag_pipeline.vector_store, 'verify_user_by_email'):
            raise HTTPException(status_code=500, detail="Authentication not available")
        
        user = rag_pipeline.vector_store.verify_user_by_email(request.email, request.password)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Get shared personalization (shared across all admins, applies to all users)
        try:
            with rag_pipeline.vector_store.SessionLocal() as db:
                admin_user = db.query(rag_pipeline.vector_store.UserORM).filter_by(role='admin').order_by(rag_pipeline.vector_store.UserORM.id.asc()).first()
                if admin_user:
                    personalization = rag_pipeline.vector_store.get_user_personalization(admin_user.id)
                    if personalization:
                        user.update(personalization)
        except Exception as e:
            logger.warning(f"Could not load personalization during login: {e}")
        
        # Create JWT token
        token = create_access_token(user["username"], user["id"])
        
        return LoginResponse(
            success=True,
            message="Login successful",
            token=token,
            user=user
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

# Registration endpoint commented out - registration disabled for now
# @app.post("/api/auth/register", response_model=LoginResponse, tags=["Authentication"])
# async def register(request: RegisterRequest):
#     """Register a new user."""
#     try:
#         if not hasattr(rag_pipeline, 'vector_store') or not hasattr(rag_pipeline.vector_store, 'create_user'):
#             raise HTTPException(status_code=500, detail="Registration not available")
#         
#         # Validate email format
#         import re
#         email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
#         if not re.match(email_pattern, request.email):
#             raise HTTPException(status_code=400, detail="Invalid email format")
#         
#         # Create user
#         user = rag_pipeline.vector_store.create_user(
#             username=request.username,
#             password=request.password,
#             email=request.email
#         )
#         
#         if not user:
#             raise HTTPException(status_code=400, detail="Username or email already exists")
#         
#         # Get personalization (will be empty for new user)
#         personalization = rag_pipeline.vector_store.get_user_personalization(user["id"])
#         if personalization:
#             user.update(personalization)
#         
#         # Create JWT token
#         token = create_access_token(user["username"], user["id"])
#         
#         return LoginResponse(
#             success=True,
#             message="Registration successful",
#             token=token,
#             user=user
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error during registration: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during registration: {str(e)}")

@app.get("/api/auth/me", tags=["Authentication"])
async def get_current_user_info(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get current authenticated user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get personalization from database and include it in user info
    # Note: Personalization is shared across all admins and applies to all users
    if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_user_personalization'):
        # Get first admin's personalization (shared across all admins, applies to all users)
        try:
            with rag_pipeline.vector_store.SessionLocal() as db:
                admin_user = db.query(rag_pipeline.vector_store.UserORM).filter_by(role='admin').order_by(rag_pipeline.vector_store.UserORM.id.asc()).first()
                if admin_user:
                    personalization = rag_pipeline.vector_store.get_user_personalization(admin_user.id)
                    if personalization:
                        current_user.update(personalization)
        except Exception as e:
            logger.warning(f"Could not load personalization: {e}")
    
    return current_user

@app.get("/api/auth/personalization", tags=["Authentication"])
async def get_personalization(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get personalization settings. Admin only - applies to all users.
    Personalization is shared across all admins."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is admin
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can access personalization")
    
    try:
        # Get first admin's personalization (shared across all admins)
        with rag_pipeline.vector_store.SessionLocal() as db:
            admin_user = db.query(rag_pipeline.vector_store.UserORM).filter_by(role='admin').order_by(rag_pipeline.vector_store.UserORM.id.asc()).first()
        
        if admin_user:
            personalization = rag_pipeline.vector_store.get_user_personalization(admin_user.id)
            if personalization:
                return personalization
        
        # Return default empty settings
        return {
            "custom_instructions": "",
            "nickname": "",
            "occupation": "",
            "more_about_you": "",
            "base_style_tone": "default"
        }
    except Exception as e:
        logger.error(f"Error getting personalization: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting personalization: {str(e)}")

@app.post("/api/auth/personalization", tags=["Authentication"])
async def save_personalization(
    personalization: Dict[str, Any],
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Save personalization settings. Admin only - applies to all users.
    Personalization is shared across all admins - any admin can change it."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is admin
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can save personalization")
    
    try:
        # Get the first admin user ID - personalization is shared across all admins
        with rag_pipeline.vector_store.SessionLocal() as db:
            admin_user = db.query(rag_pipeline.vector_store.UserORM).filter_by(role='admin').order_by(rag_pipeline.vector_store.UserORM.id.asc()).first()
            if not admin_user:
                raise HTTPException(status_code=500, detail="No admin user found")
            
            # Save to first admin's personalization (shared across all admins)
            success = rag_pipeline.vector_store.save_user_personalization(
                admin_user.id, 
                personalization
            )
        
        if success:
            return {"success": True, "message": "Personalization saved successfully. This applies to all users and admins."}
        else:
            raise HTTPException(status_code=500, detail="Failed to save personalization")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving personalization: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving personalization: {str(e)}")

@app.post("/api/auth/logout", tags=["Authentication"])
async def logout():
    """Logout endpoint (client should remove token)."""
    return {"message": "Logged out successfully"}

# User Management Endpoints (Admin Only)
# Test endpoint to verify route registration
@app.get("/api/admin/test", tags=["Admin"])
async def test_admin_endpoint():
    """Test endpoint to verify admin routes are registered."""
    return {"message": "Admin routes are working", "status": "ok"}

@app.get("/api/admin/users", tags=["Admin"])
async def get_all_users(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Get all users. Admin only."""
    logger.info("GET /api/admin/users endpoint called")
    
    if not current_user:
        logger.warning("GET /api/admin/users: Not authenticated")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is admin
    if current_user.get('role') != 'admin':
        logger.warning(f"GET /api/admin/users: User {current_user.get('username')} is not admin")
        raise HTTPException(status_code=403, detail="Only administrators can access user management")
    
    try:
        if not hasattr(rag_pipeline, 'vector_store') or not hasattr(rag_pipeline.vector_store, 'get_all_users'):
            logger.error("GET /api/admin/users: vector_store or get_all_users method not available")
            raise HTTPException(status_code=500, detail="User management not available")
        
        logger.info("GET /api/admin/users: Calling get_all_users()")
        users = rag_pipeline.vector_store.get_all_users()
        logger.info(f"GET /api/admin/users: Found {len(users)} users")
        return {"users": users, "total": len(users)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting users: {str(e)}")

@app.post("/api/admin/users", tags=["Admin"])
async def create_user_admin(
    request: RegisterRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Create a new user. Admin only."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is admin
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can create users")
    
    try:
        if not hasattr(rag_pipeline, 'vector_store') or not hasattr(rag_pipeline.vector_store, 'create_user'):
            raise HTTPException(status_code=500, detail="User management not available")
        
        user = rag_pipeline.vector_store.create_user(
            username=request.username,
            password=request.password,
            email=request.email
        )
        
        if not user:
            raise HTTPException(status_code=400, detail="Email already exists or invalid data")
        
        return {
            "success": True,
            "message": "User created successfully",
            "user": user
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.delete("/api/admin/users/{user_id}", tags=["Admin"])
async def delete_user_admin(
    user_id: int,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Delete a user. Admin only."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if user is admin
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can delete users")
    
    # Prevent admin from deleting themselves
    if current_user.get('id') == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    try:
        if not hasattr(rag_pipeline, 'vector_store') or not hasattr(rag_pipeline.vector_store, 'delete_user'):
            raise HTTPException(status_code=500, detail="User management not available")
        
        success = rag_pipeline.vector_store.delete_user(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "message": "User deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@app.post("/api/ingest", response_model=EnhancedIngestionResponse, tags=["Documents"])
async def ingest_documents_parallel(
    files: List[UploadFile] = File(..., description="PDF, CSV, or XLSX files to ingest"),
    max_workers: int = Form(default=4, description="Number of parallel workers"),
    batch_size: int = Form(default=100, description="Chunks per batch"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Ingest documents (PDF, CSV, XLSX) using parallel processing. Admin only."""
    # Check if user is admin
    if not current_user or current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can upload documents")
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types
        supported_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in supported_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a supported format. Supported: PDF, CSV, XLSX, XLS"
                )
        
        processed_docs = []
        total_chunks = 0
        
        # Process all files first
        logger.info(f"Starting document processing for {len(files)} files...")
        for file in files:
            try:
                # Ensure upload directory exists
                upload_path = Path(settings.upload_dir)
                upload_path.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded file
                file_path = upload_path / file.filename
                
                # Read file content
                file_content = await file.read()
                
                # Save to disk
                with open(file_path, "wb") as buffer:
                    buffer.write(file_content)
                
                logger.info(f"Saved file {file.filename} to {file_path}")
                
                # Process document
                doc_info = document_processor.process_document(file_path)
                processed_docs.append(doc_info)
                total_chunks += doc_info['total_chunks']
                
                logger.info(f"Processed {file.filename} into {doc_info['total_chunks']} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}", exc_info=True)
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
    files: List[UploadFile] = File(..., description="PDF, CSV, or XLSX files to ingest"),
    max_workers: int = Form(default=4),
    batch_size: int = Form(default=100),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Ingest documents asynchronously. Admin only."""
    # Check if user is admin
    if not current_user or current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can upload documents")
    logger.debug(f"Received ingestion request: {len(files)} file(s), max_workers={max_workers}, batch_size={batch_size}")
    try:
        if not files:
            logger.warning("No files provided in the request")
            raise HTTPException(status_code=400, detail="No files provided")

        saved_paths = []
        supported_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
        for file in files:
            logger.debug(f"Processing file: {file.filename}")
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in supported_extensions:
                logger.error(f"File rejected (unsupported format): {file.filename}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a supported format. Supported: PDF, CSV, XLSX, XLS"
                )

            # Ensure upload directory exists
            upload_path = Path(settings.upload_dir)
            upload_path.mkdir(parents=True, exist_ok=True)
            
            dest = upload_path / file.filename
            
            # Stream file content to disk instead of reading entire file into memory
            # This prevents memory issues and connection timeouts for large files
            try:
                with open(dest, "wb") as buffer:
                    # Read file in chunks to avoid memory issues
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = await file.read(chunk_size)
                        if not chunk:
                            break
                        buffer.write(chunk)
                
                saved_paths.append(str(dest))
                file_size = dest.stat().st_size
                logger.info(f"Saved file {file.filename} ({file_size / (1024*1024):.2f} MB) to: {dest}")
            except Exception as save_error:
                logger.error(f"Error saving file {file.filename}: {save_error}")
                # Clean up partial file if it exists
                if dest.exists():
                    try:
                        dest.unlink()
                    except:
                        pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file {file.filename}: {str(save_error)}"
                )

        logger.debug(f"Enqueuing Celery task for {len(saved_paths)} file(s)")
        
        # Check if worker is ready, wait briefly if not
        if not check_worker_ready():
            logger.info("Worker not ready, waiting 2 seconds...")
            import time
            time.sleep(2)
            
            # Check again after waiting
            if not check_worker_ready():
                logger.warning("Worker still not ready, but proceeding with task queue")
        
        task = celery_app.send_task(
            "app.tasks.ingest_documents_task",
            args=[saved_paths, max_workers, batch_size]
        )

        logger.info(f"Task queued successfully with job_id={task.id}")
        return {"job_id": task.id, "status": "queued"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error scheduling async ingestion")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ingest/progress/{job_id}", tags=["Documents"])
async def get_ingest_progress(job_id: str):
    """Get ingestion progress for a job."""
    try:
        res = AsyncResult(job_id, app=celery_app)
        
        # Check if this task is already known to be corrupted
        is_corrupted = job_id in corrupted_tasks_cache
        
        # Safely get state
        state = None
        try:
            state = res.state
            logger.debug(f"Task {job_id} state: {state}")
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            error_msg = str(e)
            if "Exception information must include the exception type" in error_msg:
                if not is_corrupted:
                    logger.warning(f"Corrupted task metadata for job {job_id}: {e}")
                    corrupted_tasks_cache.add(job_id)
                    cleanup_corrupted_task(job_id)
                    is_corrupted = True
                state = None
            else:
                logger.warning(f"Error getting state for job {job_id}: {e}")
                state = None
        
        # Build response
        response = {
            "job_id": job_id,
            "state": state or "PENDING",
            "progress": 0,
        }
        
        # If corrupted, return early
        if is_corrupted:
            response["state"] = "FAILURE"
            response["detail"] = "Task metadata is corrupted. Please start a new ingestion task."
            response["progress"] = 100
            return response
        
        # Try to get detailed info
        if state and state not in ["PENDING", "FAILURE"]:
            try:
                info = res.info
                if info and isinstance(info, dict):
                    response.update(info)
                    logger.debug(f"Task {job_id} info: {info}")
            except Exception as e:
                logger.debug(f"Could not get info for {job_id}: {e}")
        
        # Check if task is complete
        if state == "SUCCESS":
            try:
                result = res.result
                if result and isinstance(result, dict):
                    response.update(result)
                    logger.debug(f"Task {job_id} result: {result}")
            except Exception as e:
                logger.debug(f"Could not get result for {job_id}: {e}")
        
        # Check if task failed
        if state == "FAILURE":
            try:
                result = res.result
                if result and isinstance(result, dict):
                    response.update(result)
                else:
                    response["error"] = str(result) if result else "Task failed"
                logger.debug(f"Task {job_id} failure result: {result}")
            except Exception as e:
                logger.debug(f"Could not get failure result for {job_id}: {e}")
                response["error"] = "Task failed"
            response["progress"] = 100
        
        return response
        
    except Exception as e:
        logger.exception(f"Error getting progress for job {job_id}")
        return {
            "job_id": job_id,
            "state": "PENDING",
            "progress": 0,
            "detail": f"Error checking task status: {str(e)}"
        }

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest,current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Process a query through the RAG pipeline."""
    try:
        # Extract user_id before using it
        user_id = current_user.get("id") if current_user else None
        
        # Process, passing session_id to include conversational history in context
        result = rag_pipeline.process_query(request.query, request.top_k, session_id=request.session_id, user_id=user_id)
        
        # Persist chat messages if database available
        try:
            if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'save_chat_message'):
                if request.session_id:
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'user', request.query, user_id=user_id)
                    answer_text = str(result.get('answer', ''))
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'ai', answer_text, user_id=user_id)
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
async def get_chat_history(
    session_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Return stored chat history for a given session. Each user (including admins) can only see their own chats."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        # All users (including admins) can only see their own chat history
        user_id = current_user.get("id")
        
        # Verify session belongs to this user by checking if any messages exist for this session and user
        if hasattr(rag_pipeline, 'vector_store'):
            with rag_pipeline.vector_store.SessionLocal() as db:
                session_check = db.query(rag_pipeline.vector_store.ChatMessageORM).filter_by(
                    session_id=session_id,
                    user_id=user_id
                ).first()
                if not session_check:
                    raise HTTPException(status_code=403, detail="You can only access your own chat sessions")
        
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_chat_history'):
            rows = rag_pipeline.vector_store.get_chat_history(session_id, user_id=user_id)
            messages = [ChatMessage(id=r.get('id'), session_id=session_id, role=r.get('role'), message=r.get('message')) for r in rows]
            return ChatHistoryResponse(session_id=session_id, messages=messages)
        else:
            return ChatHistoryResponse(session_id=session_id, messages=[])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

@app.get("/api/chat/sessions/list", response_model=ChatSessionsListResponse, tags=["RAG"])
async def list_chat_sessions(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Return list of all chat sessions with their first message as title. Each user (including admins) can only see their own chats."""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        # All users (including admins) can only see their own chat sessions
        user_id = current_user.get("id")
        
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_all_chat_sessions'):
            sessions_data = rag_pipeline.vector_store.get_all_chat_sessions(user_id=user_id)
            sessions = [
                ChatSessionInfo(
                    session_id=s.get('session_id'),
                    title=s.get('title', 'New Chat'),
                    created_at=s.get('created_at', 0)
                )
                for s in sessions_data
            ]
            return ChatSessionsListResponse(sessions=sessions, total_sessions=len(sessions))
        else:
            return ChatSessionsListResponse(sessions=[], total_sessions=0)
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing chat sessions: {str(e)}")

@app.delete("/api/chat/sessions/{session_id}", tags=["RAG"])
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages."""
    try:
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'delete_chat_session'):
            success = rag_pipeline.vector_store.delete_chat_session(session_id)
            if success:
                return {"message": "Chat session deleted successfully", "session_id": session_id}
            else:
                raise HTTPException(status_code=404, detail="Chat session not found or already deleted")
        else:
            raise HTTPException(status_code=500, detail="Chat session deletion not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat session: {str(e)}")

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

@app.delete("/api/documents/{filename:path}", tags=["Documents"])
async def delete_document(
    filename: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Delete a single document and all its chunks from the knowledge base. Admin only."""
    # Debug logging
    logger.info(f"DELETE /api/documents/{filename} - current_user: {current_user}")
    if current_user:
        logger.info(f"User role: {current_user.get('role')}, User ID: {current_user.get('id')}, Username: {current_user.get('username')}")
    
    # Check if user is admin (case-insensitive check)
    if not current_user:
        logger.warning(f"DELETE /api/documents/{filename} - No current_user (not authenticated)")
        raise HTTPException(status_code=403, detail="Only administrators can delete documents")
    
    user_role = current_user.get('role')
    # Case-insensitive role check
    if not user_role or str(user_role).lower() != 'admin':
        logger.warning(f"DELETE /api/documents/{filename} - User {current_user.get('username')} (ID: {current_user.get('id')}) has role '{user_role}', not 'admin'")
        raise HTTPException(status_code=403, detail="Only administrators can delete documents")
    try:
        # URL decode the filename in case it has special characters
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        success = rag_pipeline.delete_document(filename)
        if success:
            return {"message": f"Document '{filename}' deleted successfully", "filename": filename}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found or already deleted")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

def scrape_url(url: str, depth: int = 1, visited=None, use_proxy: bool = False, proxy_list: List[str] = None):
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)
    results = []

    # Create robust session with retry mechanism
    session = requests.Session()

    # Rotate between different realistic user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ]

    # Enhanced headers with more realistic browser behavior
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
    })

    session.max_redirects = 5
    session.cookies.clear()

    # Set a realistic referer for the domain
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc:
            session.headers['Referer'] = f"https://{parsed_url.netloc}/"
    except Exception:
        pass

    # Configure proxy if enabled
    if use_proxy and proxy_list:
        try:
            proxy = random.choice(proxy_list)
            session.proxies = {'http': proxy, 'https': proxy}
            logger.info(f"Using proxy: {proxy}")
        except Exception as e:
            logger.warning(f"Failed to configure proxy: {e}")

    # Retry mechanism with exponential backoff
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            delay = random.uniform(base_delay, base_delay + 2) + (attempt * 2)
            time.sleep(delay)

            if attempt > 0:
                session.headers['User-Agent'] = random.choice(user_agents)
                session.headers['Accept-Language'] = random.choice([
                    'en-US,en;q=0.9',
                    'en-US,en;q=0.9,es;q=0.8',
                    'en-GB,en;q=0.9,en-US;q=0.8'
                ])

            res = session.get(url, timeout=20, allow_redirects=True)

            if res.status_code == 403:
                logger.warning(f"403 Forbidden for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                results.append({"url": url, "error": f"403 Forbidden after {max_retries} attempts"})
                return results

            elif res.status_code == 429:
                logger.warning(f"Rate limited for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(10 + (attempt * 5))
                    continue
                results.append({"url": url, "error": f"Rate limited after {max_retries} attempts"})
                return results

            elif res.status_code == 200:
                res.raise_for_status()
                break
            else:
                res.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            results.append({"url": url, "error": f"Request failed after {max_retries} attempts: {str(e)}"})
            return results

    # If we get here, request was successful
    try:
        soup = BeautifulSoup(res.text, "html.parser")

        for unwanted in soup(["script", "style", "noscript"]):
            unwanted.decompose()

        for unwanted in soup(["nav", "header", "footer", "aside", "advertisement", "ads",
                              "social-share", "cookie-banner", "iframe"]):
            text_content = unwanted.get_text(strip=True)
            if text_content and len(text_content) > 20:
                temp_div = soup.new_tag("div", **{"class": "extracted-ui-text", "data-source": unwanted.name})
                temp_div.string = f"UI_ELEMENT_{unwanted.name.upper()}: {text_content}"
                soup.append(temp_div)
            unwanted.decompose()

        # Extract different elements
        headings, heading_hierarchy, paragraphs, div_content = [], [], [], []
        semantic_content, list_content, table_content = [], [], []
        form_content, span_content, code_content = [], [], []
        other_content, data_content, meta_content, selector_content = [], [], [], []
        remaining_text = []

        # Headings
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            heading_text = h.get_text(strip=True)
            if heading_text and len(heading_text) > 2:
                headings.append(heading_text)
                level = int(h.name[1])
                heading_hierarchy.append(f"{'  ' * (level-1)}• {heading_text}")

        # Paragraphs
        for p in soup.find_all("p"):
            para_text = p.get_text(strip=True)
            if para_text and len(para_text) > 10:
                paragraphs.append(para_text)

        # Divs
        for div in soup.find_all("div"):
            div_class = " ".join(div.get("class", [])).lower()
            div_id = div.get("id", "").lower()
            if any(skip_word in div_class or skip_word in div_id for skip_word in
                   ["nav", "menu", "sidebar", "ad", "advertisement", "banner", "footer",
                    "header", "social", "share", "comment", "cookie", "popup", "modal"]):
                continue
            div_text = div.get_text(strip=True)
            if div_text and len(div_text) > 20:
                div_content.append(div_text)

        # Semantic
        for tag in soup.find_all(["main", "article", "section", "aside", "details", "summary",
                                  "blockquote", "cite", "q", "mark", "ins", "del", "s", "u"]):
            content_text = tag.get_text(strip=True)
            if content_text and len(content_text) > 15:
                semantic_content.append(content_text)

        # Lists
        for list_tag in soup.find_all(["ul", "ol", "dl"]):
            list_text = list_tag.get_text(strip=True)
            if list_text and len(list_text) > 20:
                list_content.append(list_text)

        # Tables
        for table in soup.find_all("table"):
            table_text = table.get_text(strip=True)
            if table_text and len(table_text) > 30:
                table_content.append(table_text)

        # Forms
        for form in soup.find_all("form"):
            form_text = form.get_text(strip=True)
            if form_text and len(form_text) > 20:
                form_content.append(form_text)

        # Spans
        for span in soup.find_all("span"):
            span_text = span.get_text(strip=True)
            if span_text and len(span_text) > 10:
                span_content.append(span_text)

        # Code blocks
        for code_tag in soup.find_all(["pre", "code", "kbd", "samp", "var"]):
            code_text = code_tag.get_text(strip=True)
            if code_text and len(code_text) > 10:
                code_content.append(code_text)

        # Other text
        for tag in soup.find_all(["strong", "b", "em", "i", "small", "sub", "sup",
                                  "abbr", "acronym", "address", "bdo", "big", "tt"]):
            tag_text = tag.get_text(strip=True)
            if tag_text and len(tag_text) > 5:
                other_content.append(tag_text)

        # Data attributes
        for element in soup.find_all(attrs={"data-content": True}):
            data_text = element.get("data-content", "").strip()
            if data_text and len(data_text) > 10:
                data_content.append(data_text)

        # Title + meta
        title = soup.find('title')
        if title and title.get_text(strip=True):
            meta_content.append(f"Page Title: {title.get_text(strip=True)}")

        for meta in soup.find_all('meta', attrs={'name': ['description', 'keywords', 'author', 'subject']}):
            content = meta.get('content', '').strip()
            if content:
                meta_content.append(f"{meta.get('name', 'meta')}: {content}")

        # Content selectors
        content_selectors = [
            '.content', '.main-content', '.article-content', '.post-content',
            '.entry-content', '.page-content', '.text-content', '.body-content',
            '.content-body', '.main-body', '.article-body', '.post-body',
            '.content-area', '.main-area', '.primary-content', '.secondary-content',
            '[role="main"]', '[role="article"]', '[role="contentinfo"]',
            '.container', '.wrapper', '.inner', '.content-wrapper'
        ]
        for selector in content_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        selector_content.append(f"SELECTOR_{selector}: {text}")
            except Exception:
                continue

        # Remaining text
        for element in soup.find_all(text=True):
            if element.parent and element.parent.name not in ['script', 'style', 'noscript']:
                text = element.strip()
                if text and len(text) > 5:
                    remaining_text.append(text)

        # Organize by priority
        high_priority_content = [
            ("Headings", headings),
            ("Paragraphs", paragraphs),
            ("Semantic Content", semantic_content),
            ("Table Content", table_content),
            ("Content Selectors", selector_content)
        ]
        medium_priority_content = [
            ("List Content", list_content),
            ("Code Content", code_content),
            ("Meta Information", meta_content),
            ("Data Attributes", data_content)
        ]
        low_priority_content = [
            ("Div Content", div_content),
            ("Form Content", form_content),
            ("Span Content", span_content),
            ("Other Text Elements", other_content),
            ("Remaining Text", remaining_text)
        ]

        all_text_parts = []
        for priority_section, content_groups in [
            ("HIGH PRIORITY CONTENT", high_priority_content),
            ("MEDIUM PRIORITY CONTENT", medium_priority_content),
            ("LOW PRIORITY CONTENT", low_priority_content)
        ]:
            all_text_parts.append(f"\n{'='*20} {priority_section} {'='*20}")
            for content_name, content_list in content_groups:
                if content_list:
                    all_text_parts.append(f"\n--- {content_name.upper()} ---")
                    all_text_parts.extend(content_list)

        all_text = "\n".join(all_text_parts)

        # Cleanup
        all_text = re.sub(r'\n\s*\n', '\n\n', all_text)
        all_text = re.sub(r'[ \t]+', ' ', all_text)
        for pattern in [
            r'Cookie Policy|Privacy Policy|Terms of Service|Subscribe|Newsletter|Sign up|Login|Register',
            r'From Wikipedia, the free encyclopedia|Jump to navigation|Jump to search',
            r'Skip to main content|Skip to navigation|Skip to search',
            r'Home|About|Contact|Support|Help|FAQ',
            r'Follow us|Share|Like|Tweet|Share on|Follow on',
            r'Advertisement|Ad|Sponsored|Promoted',
            r'© \d{4}|All rights reserved|Copyright',
            r'Last updated|Last modified|Published on|Posted on',
            r'Read more|Continue reading|Show more|View more',
            r'Loading|Please wait|Error|Warning|Notice',
            r'JavaScript is required|Enable JavaScript|Update your browser',
            r'This site uses cookies|Accept cookies|Cookie settings'
        ]:
            all_text = re.sub(pattern, '', all_text, flags=re.IGNORECASE)

        all_text = re.sub(r'\n{3,}', '\n\n', all_text).strip()

        if all_text and len(all_text) > 50:
            results.append({
                "url": url,
                "headings": headings,
                "text": all_text
            })
            logger.info(f"Successfully scraped {url}: {len(all_text)} chars, {len(headings)} headings")
        else:
            logger.warning(f"Insufficient content scraped from {url}")

        # Depth handling
        if depth > 1:
            base_domain = urlparse(url).netloc
            links = soup.find_all("a", href=True)
            followed_links = 0
            for a in links:
                if followed_links >= 5:
                    break
                child_url = urljoin(url, a["href"])
                if (child_url.startswith("http") and
                        urlparse(child_url).netloc == base_domain and
                        not any(child_url.lower().endswith(ext) for ext in
                                ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.xml', '.json'])):
                    try:
                        child_results = scrape_url(child_url, depth - 1, visited)
                        results.extend(child_results)
                        followed_links += 1
                    except Exception as e:
                        logger.warning(f"Failed to scrape child URL {child_url}: {e}")
                        continue

    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        results.append({"url": url, "error": str(e)})
    finally:
        session.close()

    return results


@app.post("/api/weblink", tags=["Weblink"])
async def weblink_endpoint(
    request: Dict[str, Any],
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Schedule weblink scraping and ingestion as a background Celery task.
    Returns a job_id to poll with any Celery-aware progress endpoint.
    Admin only.
    
    Optional parameters:
    - use_proxy: bool - Enable proxy usage for scraping
    - proxy_list: List[str] - List of proxy URLs to use
    """
    # Check if user is admin
    if not current_user or current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Only administrators can upload web links")
    
    try:
        url = request.get("query")
        depth = int(request.get("depth", 1))
        max_workers = int(request.get("max_workers", 4))
        batch_size = int(request.get("batch_size", 100))
        use_proxy = request.get("use_proxy", False)
        proxy_list = request.get("proxy_list", [])

        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")

        logger.debug(f"Enqueuing Celery task for weblink ingestion")
        
        # Check if worker is ready
        if not check_worker_ready():
            logger.info("Worker not ready, waiting 2 seconds...")
            import time
            time.sleep(2)
        
        task = celery_app.send_task(
            "app.tasks.weblink_ingestion_task",
            args=[url, depth, max_workers, batch_size, use_proxy, proxy_list]
        )
        return {"job_id": task.id, "status": "queued", "use_proxy": use_proxy}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling weblink async ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scrape-test", tags=["System"])
async def test_scraping(url: str = "https://httpbin.org/user-agent", use_proxy: bool = False):
    """Test web scraping functionality with a simple URL."""
    try:
        result = scrape_url(url, depth=1, use_proxy=use_proxy)
        
        if result and len(result) > 0:
            if 'error' in result[0]:
                return {
                    "success": False,
                    "error": result[0]['error'],
                    "url": url,
                    "use_proxy": use_proxy
                }
            else:
                return {
                    "success": True,
                    "url": url,
                    "content_length": len(result[0].get('text', '')),
                    "headings_count": len(result[0].get('headings', [])),
                    "use_proxy": use_proxy,
                    "preview": result[0].get('text', '')[:200] + "..." if len(result[0].get('text', '')) > 200 else result[0].get('text', '')
                }
        else:
            return {
                "success": False,
                "error": "No content scraped",
                "url": url,
                "use_proxy": use_proxy
            }
            
    except Exception as e:
        logger.error(f"Error testing scraping: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "use_proxy": use_proxy
        }

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
            model="unknown",
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