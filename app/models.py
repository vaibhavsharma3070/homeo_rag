from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Client session identifier for chat continuity")

class QueryResponse(BaseModel):
    """Response model for query processing."""
    query: str
    answer: str
    context_used: List[str]
    sources: List[Dict[str, Any]]
    confidence: str
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class SearchResponse(BaseModel):
    """Response model for document search."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentInfo(BaseModel):
    """Model for document information."""
    id: str
    filename: str
    file_path: str
    total_chunks: int
    metadata: Dict[str, Any]

class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentInfo]
    total_documents: int
    timestamp: datetime = Field(default_factory=datetime.now)

class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    timestamp: datetime = Field(default_factory=datetime.now)

class StatsResponse(BaseModel):
    """Response model for knowledge base statistics."""
    total_documents: int
    total_chunks: int
    index_size: int
    embedding_dimension: int
    llm_available: bool
    llm_provider: str
    timestamp: datetime = Field(default_factory=datetime.now)

class LLMTestResponse(BaseModel):
    """Response model for LLM connection test."""
    status: str
    provider: str
    model:str
    test_successful: Optional[bool] = None
    test_response: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSessionResponse(BaseModel):
    """Response model for creating a new chat session."""
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatMessage(BaseModel):
    """A single chat message."""
    id: Optional[int] = None
    session_id: str
    role: str
    message: str
    created_at: datetime = Field(default_factory=datetime.now)

class ChatHistoryResponse(BaseModel):
    """Response containing chat history for a session."""
    session_id: str
    messages: List[ChatMessage]
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSessionInfo(BaseModel):
    """Information about a chat session."""
    session_id: str
    title: str
    created_at: int

class ChatSessionsListResponse(BaseModel):
    """Response containing list of all chat sessions."""
    sessions: List[ChatSessionInfo]
    total_sessions: int
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., min_length=1, max_length=100, description="Username")
    password: str = Field(..., min_length=1, description="Password")

class LoginResponse(BaseModel):
    """Response model for user login."""
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class UserInfo(BaseModel):
    """User information model."""
    id: int
    username: str
    created_at: int