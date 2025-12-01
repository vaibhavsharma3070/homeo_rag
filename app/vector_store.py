import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.config import settings
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import time
from dataclasses import dataclass

try:
    # LangChain imports
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_postgres import PGVector
    from langchain_core.documents import Document as LangChainDocument
    
    # PostgreSQL and pgvector
    from sqlalchemy import Column, Integer, String, Text, JSON, create_engine, select, func, text
    from sqlalchemy.orm import declarative_base, Session, sessionmaker
    from sqlalchemy.pool import QueuePool
    from pgvector.sqlalchemy import Vector
    import pgvector
    PGVECTOR_AVAILABLE = True
except Exception as e:
    PGVECTOR_AVAILABLE = False
    print(f"pgvector not available: {e}")

@dataclass
class ChunkBatch:
    """Represents a batch of chunks to be processed"""
    batch_id: int
    chunks: List[Dict[str, Any]]
    document_info: Dict[str, Any]

@dataclass
class IngestionProgress:
    """Track ingestion progress"""
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    start_time: float = 0
    current_batch: int = 0
    total_batches: int = 0

Base = None

class PGVectorStore:
    """Enhanced PostgreSQL + pgvector-based vector store with LangChain integration."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        if not PGVECTOR_AVAILABLE:
            raise RuntimeError("pgvector dependencies not installed. Install psycopg2, sqlalchemy, pgvector.")
        
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.progress = IngestionProgress()
        self.progress_lock = threading.Lock()
        
        # Initialize embedding model (shared across workers)
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Create engine with connection pooling for parallel access
        self.engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=max_workers * 2,
            max_overflow=max_workers,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self._init_pg_base()
        self._setup_database()
        
        # Initialize LangChain PGVector store
        self._init_langchain_vectorstore()
    
    def close(self):
        """Close database connection and cleanup resources."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    def _init_langchain_vectorstore(self):
        """Initialize LangChain's PGVector store."""
        # Create LangChain-compatible embeddings
        self.langchain_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Initialize PGVector store with LangChain's default tables
        self.vectorstore = PGVector(
            embeddings=self.langchain_embeddings,
            connection=settings.database_url,
            use_jsonb=True,
        )
        
        logger.info("LangChain PGVector store initialized")

    def _init_pg_base(self):
        global Base
        if Base is None:
            from sqlalchemy.orm import declarative_base as _declarative_base
            Base = _declarative_base()
        
        # Keep old table for chat messages
        class ChatMessageORM(Base):
            __tablename__ = "chat_messages"
            __table_args__ = {"extend_existing": True}

            id = Column(Integer, primary_key=True, autoincrement=True)
            session_id = Column(String(64), index=True, nullable=False)
            user_id = Column(Integer, nullable=True, index=True)  # Track which user owns this chat
            role = Column(String(16), nullable=False)  # 'user' or 'ai' (message role)
            message = Column(Text, nullable=False)
            embedding = Column(Vector(self.dimension))
            created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))

        self.ChatMessageORM = ChatMessageORM

        # User authentication table
        class UserORM(Base):
            __tablename__ = "users"
            __table_args__ = {"extend_existing": True}

            id = Column(Integer, primary_key=True, autoincrement=True)
            username = Column(String(100), unique=True, nullable=False, index=True)
            email = Column(String(255), unique=True, nullable=True, index=True)
            password = Column(String(255), nullable=False)  # Will store hashed password
            role = Column(String(16), nullable=False, default='user')  # 'user' or 'admin'
            created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))

        self.UserORM = UserORM

        class UserPersonalizationORM(Base):
            __tablename__ = "user_personalization"
            __table_args__ = {"extend_existing": True}

            id = Column(Integer, primary_key=True, autoincrement=True)
            user_id = Column(Integer, nullable=False, unique=True, index=True)
            custom_instructions = Column(Text)
            nickname = Column(String(100))
            occupation = Column(String(200))
            more_about_you = Column(Text)
            base_style_tone = Column(String(50), default='default')
            updated_at = Column(Integer, nullable=False, default=lambda: int(time.time()))

        self.UserPersonalizationORM = UserPersonalizationORM

    def _setup_database(self):
        """Setup database with vector extension and tables."""
        try:
            with self.SessionLocal() as session:
                # Enable vector extension
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                session.commit()
                logger.info("Vector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable vector extension: {e}")
        
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")
        
        # Migration: Add email column to users table if it doesn't exist
        try:
            with self.SessionLocal() as session:
                # Check if email column exists
                result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='users' AND column_name='email'
                """))
                if result.fetchone() is None:
                    # Add email column (nullable to allow existing rows)
                    session.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR(255)"))
                    # Create unique index on email (PostgreSQL allows multiple NULLs in unique indexes)
                    session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users(email)"))
                    session.commit()
                    logger.info("Added email column to users table")
        except Exception as e:
            logger.warning(f"Could not add email column (may already exist): {e}")
        
        # Migration: Add role column to users table if it doesn't exist
        try:
            with self.SessionLocal() as session:
                result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='users' AND column_name='role'
                """))
                if result.fetchone() is None:
                    # Add role column with default 'user'
                    session.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(16) DEFAULT 'user'"))
                    session.execute(text("UPDATE users SET role = 'user' WHERE role IS NULL"))
                    session.execute(text("ALTER TABLE users ALTER COLUMN role SET NOT NULL"))
                    session.commit()
                    logger.info("Added role column to users table")
        except Exception as e:
            logger.warning(f"Could not add role column (may already exist): {e}")
        
        # Migration: Add user_id column to chat_messages table if it doesn't exist
        try:
            with self.SessionLocal() as session:
                result = session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='chat_messages' AND column_name='user_id'
                """))
                if result.fetchone() is None:
                    # Add user_id column (nullable for existing messages)
                    session.execute(text("ALTER TABLE chat_messages ADD COLUMN user_id INTEGER"))
                    session.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_messages_user_id ON chat_messages(user_id)"))
                    session.commit()
                    logger.info("Added user_id column to chat_messages table")
        except Exception as e:
            logger.warning(f"Could not add user_id column (may already exist): {e}")

    def save_chat_message(self, session_id: str, role: str, message: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Persist a chat message with optional embedding."""
        if role not in ("user", "ai"):
            role = "ai"
        vector = None
        try:
            vector = self.embedding_model.encode([message])[0].astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to embed chat message: {e}")
            vector = None

        with self.SessionLocal() as db:
            obj = self.ChatMessageORM(
                session_id=session_id,
                user_id=user_id,
                role=role,
                message=message,
                embedding=vector
            )
            db.add(obj)
            db.commit()
            db.refresh(obj)
            return {
                "id": obj.id,
                "session_id": obj.session_id,
                "user_id": obj.user_id,
                "role": obj.role,
                "message": obj.message,
                "created_at": obj.created_at
            }

    def get_chat_history(self, session_id: str, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch chat messages for a session ordered by created_at ascending.
        If user_id is provided and user is not admin, only returns messages for that user."""
        with self.SessionLocal() as db:
            query = db.query(self.ChatMessageORM).filter_by(session_id=session_id)
            # Filter by user_id if provided (for non-admin users) - only show their own messages
            if user_id is not None:
                query = query.filter(self.ChatMessageORM.user_id == user_id)
            rows = query.order_by(
                self.ChatMessageORM.created_at.asc(), 
                self.ChatMessageORM.id.asc()
            ).all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "role": r.role,
                    "message": r.message,
                    "created_at": r.created_at
                }
                for r in rows
            ]

    def save_user_personalization(self, user_id: int, personalization: Dict[str, Any]) -> bool:
        """Save or update user personalization settings."""
        try:
            with self.SessionLocal() as db:
                existing = db.query(self.UserPersonalizationORM).filter_by(user_id=user_id).first()
                
                if existing:
                    # Update existing
                    existing.custom_instructions = personalization.get('custom_instructions', '')
                    existing.nickname = personalization.get('nickname', '')
                    existing.occupation = personalization.get('occupation', '')
                    existing.more_about_you = personalization.get('more_about_you', '')
                    existing.base_style_tone = personalization.get('base_style_tone', 'default')
                    existing.updated_at = int(time.time())
                else:
                    # Create new
                    new_pref = self.UserPersonalizationORM(
                        user_id=user_id,
                        custom_instructions=personalization.get('custom_instructions', ''),
                        nickname=personalization.get('nickname', ''),
                        occupation=personalization.get('occupation', ''),
                        more_about_you=personalization.get('more_about_you', ''),
                        base_style_tone=personalization.get('base_style_tone', 'default'),
                        updated_at=int(time.time())
                    )
                    db.add(new_pref)
                
                db.commit()
                logger.info(f"Saved personalization for user {user_id}")
                print(f"✅ Personalization details SAVED for user_id={user_id}: {personalization}")
                return True
        except Exception as e:
            logger.error(f"Error saving personalization: {e}")
            return False

    def get_user_personalization(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user personalization settings."""
        try:
            with self.SessionLocal() as db:
                pref = db.query(self.UserPersonalizationORM).filter_by(user_id=user_id).first()
                
                if pref:
                    personalization_data = {
                        "custom_instructions": pref.custom_instructions or "",
                        "nickname": pref.nickname or "",
                        "occupation": pref.occupation or "",
                        "more_about_you": pref.more_about_you or "",
                        "base_style_tone": pref.base_style_tone or "default",
                        "updated_at": pref.updated_at
                    }
                    print(f"📥 Personalization details FETCHED for user_id={user_id}: {personalization_data}")
                    return personalization_data
                print(f"⚠️ No personalization found for user_id={user_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting personalization: {e}")
            return None

    def get_all_chat_sessions(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all chat sessions with their first user message as title, ordered by most recent.
        If user_id is provided and user is not admin, only returns sessions for that user."""
        with self.SessionLocal() as db:
            # Get distinct session_ids with their first user message
            sessions_query = db.query(
                self.ChatMessageORM.session_id,
                func.min(self.ChatMessageORM.created_at).label('first_message_time'),
                func.min(self.ChatMessageORM.id).label('first_message_id')
            )
            
            # Filter by user_id if provided (for non-admin users) - only show their own sessions
            if user_id is not None:
                sessions_query = sessions_query.filter(self.ChatMessageORM.user_id == user_id)
            
            sessions_query = sessions_query.group_by(self.ChatMessageORM.session_id).order_by(
                func.min(self.ChatMessageORM.created_at).desc()
            ).all()
            
            sessions = []
            for session_id, first_time, first_id in sessions_query:
                # Get the first user message for this session
                first_user_msg_query = db.query(self.ChatMessageORM).filter_by(
                    session_id=session_id,
                    role='user'
                )
                
                # Filter by user_id if provided - only show their own messages
                if user_id is not None:
                    first_user_msg_query = first_user_msg_query.filter(self.ChatMessageORM.user_id == user_id)
                
                first_user_msg = first_user_msg_query.order_by(
                    self.ChatMessageORM.created_at.asc(),
                    self.ChatMessageORM.id.asc()
                ).first()
                
                title = first_user_msg.message[:100] + "..." if first_user_msg and len(first_user_msg.message) > 100 else (first_user_msg.message if first_user_msg else "New Chat")
                
                sessions.append({
                    "session_id": session_id,
                    "title": title,
                    "created_at": first_time,
                    "first_message_id": first_id
                })
            
            return sessions

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete all chat messages for a given session."""
        try:
            with self.SessionLocal() as db:
                deleted_count = db.query(self.ChatMessageORM).filter_by(session_id=session_id).delete()
                db.commit()
                logger.info(f"Deleted {deleted_count} messages for session {session_id}")
                return deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting chat session {session_id}: {e}")
            return False

    def verify_user_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials by username and return user info if valid."""
        import hashlib
        try:
            with self.SessionLocal() as db:
                user = db.query(self.UserORM).filter_by(username=username).first()
                if not user:
                    return None
                
                # Hash the provided password and compare
                # Using SHA256 for simplicity (in production, use bcrypt)
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                if user.password == password_hash:
                    return {
                        "id": user.id,
                        "username": user.username,
                        "email": getattr(user, 'email', None),
                        "role": getattr(user, 'role', 'user'),
                        "created_at": user.created_at
                    }
                return None
        except Exception as e:
            logger.error(f"Error verifying user credentials: {e}")
            return None

    def verify_user_by_email(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials by email and return user info if valid."""
        import hashlib
        try:
            with self.SessionLocal() as db:
                user = db.query(self.UserORM).filter_by(email=email).first()
                if not user:
                    return None
                
                # Hash the provided password and compare
                # Using SHA256 for simplicity (in production, use bcrypt)
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                if user.password == password_hash:
                    return {
                        "id": user.id,
                        "username": user.username,
                        "email": getattr(user, 'email', None),
                        "role": getattr(user, 'role', 'user'),
                        "created_at": user.created_at
                    }
                return None
        except Exception as e:
            logger.error(f"Error verifying user credentials by email: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information by username."""
        try:
            with self.SessionLocal() as db:
                user = db.query(self.UserORM).filter_by(username=username).first()
                if user:
                    return {
                        "id": user.id,
                        "username": user.username,
                        "email": getattr(user, 'email', None),
                        "role": getattr(user, 'role', 'user'),
                        "created_at": user.created_at
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None

    def create_user(self, username: str, password: str, email: Optional[str] = None, role: str = 'user') -> Optional[Dict[str, Any]]:
        """Create a new user (for admin use)."""
        import hashlib
        try:
            with self.SessionLocal() as db:
                # Check if user already exists
                existing = db.query(self.UserORM).filter_by(username=username).first()
                if existing:
                    return None
                
                # Check if email already exists (if provided)
                if email:
                    existing_email = db.query(self.UserORM).filter_by(email=email).first()
                    if existing_email:
                        return None
                
                # Hash password
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                user = self.UserORM(
                    username=username,
                    email=email,
                    password=password_hash,
                    role=role  # Role for new users (default: 'user')
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": getattr(user, 'role', 'user'),
                    "created_at": user.created_at
                }
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users (for admin use)."""
        try:
            with self.SessionLocal() as db:
                users = db.query(self.UserORM).all()
                return [
                    {
                        "id": user.id,
                        "username": user.username,
                        "email": getattr(user, 'email', None),
                        "role": getattr(user, 'role', 'user'),
                        "created_at": user.created_at
                    }
                    for user in users
                ]
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID (for admin use)."""
        try:
            with self.SessionLocal() as db:
                user = db.query(self.UserORM).filter_by(id=user_id).first()
                if not user:
                    return False
                
                # Prevent deleting admin users (optional safety check)
                if getattr(user, 'role', 'user') == 'admin':
                    # Allow deleting admin but log it
                    logger.warning(f"Admin user {user.username} (ID: {user_id}) is being deleted")
                
                db.delete(user)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}")
            return False

    def _create_chunk_batches(self, documents: List[Dict[str, Any]]) -> List[ChunkBatch]:
        """Create batches with complete metadata."""
        batches = []
        batch_id = 0
        
        for doc in documents:
            chunks = doc['chunks']
            total_chunks = len(chunks)
            chunk_metadata_list = doc.get('chunk_metadata', [])
            
            for i in range(0, total_chunks, self.batch_size):
                batch_chunks = []
                batch_end = min(i + self.batch_size, total_chunks)
                
                for chunk_idx in range(i, batch_end):
                    chunk_data = {
                        'filename': doc['filename'],
                        'file_path': doc['file_path'],
                        'chunk_index': chunk_idx,
                        'text': chunks[chunk_idx],
                        'document_metadata': {
                        **doc.get('metadata', {}),
                        'chunk_metadata': chunk_metadata_list[chunk_idx] if chunk_metadata_list and len(chunk_metadata_list) > chunk_idx else {}
                    },

                        'total_chunks': total_chunks
                    }
                    batch_chunks.append(chunk_data)
                
                batches.append(ChunkBatch(
                    batch_id=batch_id,
                    chunks=batch_chunks,
                    document_info={
                        'filename': doc['filename'],
                        'file_path': doc['file_path'],
                        'total_chunks': total_chunks
                    }
                ))
                batch_id += 1
        
        return batches

    def _process_chunk_batch(self, batch: ChunkBatch) -> Dict[str, Any]:
        """Process a single batch of chunks and store with RICH metadata."""
        try:
            langchain_docs = []
            
            for chunk in batch.chunks:
                # Get source type
                source_type = chunk.get('source_type', 'pdf')
                if 'source_url' in chunk.get('document_metadata', {}):
                    source_type = 'url'
                elif chunk.get('file_path', '').startswith('http'):
                    source_type = 'url'
                
                # Base metadata
                metadata = {
                    'filename': chunk['filename'],
                    'file_path': chunk['file_path'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'source': source_type,
                    'created_at': int(time.time()),
                }
                
                # CRITICAL: Add chunk-specific metadata if available
                doc_metadata = chunk.get('document_metadata', {})
                chunk_metadata_list = doc_metadata.get('chunk_metadata', [])
                
                # If we have rich metadata for this chunk, add it
                if chunk_metadata_list and chunk['chunk_index'] < len(chunk_metadata_list):
                    chunk_meta = chunk_metadata_list[chunk['chunk_index']]
                    
                    # Add row number
                    if 'row_number' in chunk_meta:
                        metadata['row_number'] = chunk_meta['row_number']
                    
                    # Add all searchable fields from the row
                    for key, value in chunk_meta.items():
                        if key.startswith('field_'):
                            metadata[key] = value
                    
                    # Store complete row data for exact retrieval
                    if 'row_data' in chunk_meta:
                        metadata['row_data'] = str(chunk_meta['row_data'])
                
                # Add other document metadata
                for k, v in doc_metadata.items():
                    if k not in ['chunk_metadata'] and k not in metadata:
                        metadata[k] = v
                
                # Create LangChain document
                doc = LangChainDocument(
                    page_content=chunk['text'],
                    metadata=metadata
                )
                langchain_docs.append(doc)
            
            # Add to vectorstore
            self.vectorstore.add_documents(langchain_docs)
            
            # Update progress
            with self.progress_lock:
                self.progress.processed_chunks += len(batch.chunks)
                self.progress.current_batch += 1
                
                elapsed = time.time() - self.progress.start_time
                chunks_per_second = self.progress.processed_chunks / elapsed if elapsed > 0 else 0
                
                logger.info(
                    f"Batch {batch.batch_id}: {self.progress.processed_chunks}/{self.progress.total_chunks} "
                    f"({chunks_per_second:.1f} chunks/sec)"
                )
            
            return {
                'batch_id': batch.batch_id,
                'status': 'success',
                'chunks_processed': len(batch.chunks),
                'filename': batch.document_info['filename']
            }
            
        except Exception as e:
            with self.progress_lock:
                self.progress.failed_chunks += len(batch.chunks)
            logger.error(f"Batch {batch.batch_id} error: {e}")
            return {
                'batch_id': batch.batch_id,
                'status': 'error',
                'error': str(e),
                'chunks_failed': len(batch.chunks)
            }

    def add_documents_parallel(self, documents: List[Dict[str, Any]], 
                             progress_callback=None) -> Dict[str, Any]:
        """Add documents using parallel processing with LangChain."""
        start_time = time.time()
        
        total_chunks = sum(doc['total_chunks'] for doc in documents)
        self.progress = IngestionProgress(
            total_chunks=total_chunks,
            start_time=start_time
        )
        
        batches = self._create_chunk_batches(documents)
        self.progress.total_batches = len(batches)
        
        logger.info(f"Starting parallel ingestion: {len(documents)} documents, "
                   f"{total_chunks} chunks, {len(batches)} batches, {self.max_workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_chunk_batch, batch): batch 
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_info = {
                            'processed_chunks': self.progress.processed_chunks,
                            'total_chunks': self.progress.total_chunks,
                            'current_batch': self.progress.current_batch,
                            'total_batches': self.progress.total_batches,
                            'elapsed_time': time.time() - start_time
                        }
                        progress_callback(progress_info)
                        
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    results.append({
                        'batch_id': batch.batch_id,
                        'status': 'error',
                        'error': str(e)
                    })
        
        successful_batches = sum(1 for r in results if r['status'] == 'success')
        failed_batches = len(results) - successful_batches
        total_time = time.time() - start_time
        
        final_stats = {
            'success': failed_batches == 0,
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'processed_chunks': self.progress.processed_chunks,
            'failed_chunks': self.progress.failed_chunks,
            'total_batches': len(batches),
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'processing_time': total_time,
            'chunks_per_second': total_chunks / total_time if total_time > 0 else 0,
            'batch_results': results
        }
        
        logger.info(f"Parallel ingestion completed: {successful_batches}/{len(batches)} batches successful, "
                   f"{total_time:.2f}s total, {final_stats['chunks_per_second']:.1f} chunks/sec")
        
        return final_stats

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents (uses parallel processing internally)."""
        result = self.add_documents_parallel(documents)
        return result['success']

    def search_with_agent(self, query: str, history: List[Dict[str, str]] = None, user_id: Optional[int] = None) -> Optional[str]:
        """Search using the intelligent agent with conversation history."""
        try:
            from app.agent import run_agent
            logger.info(f"Attempting agent search for: '{query}'")
            print('query =========================================== ',query)
            print('history =========================================== ',history)
            result = run_agent(query, history=history or [], max_iterations=5, user_id=user_id)
            print('vector result =========================================== ',result)
            
            # Check if result exists and is valid
            if not result:
                logger.info("Agent returned empty result")
                return None
            
            result_lower = result.lower()
            
            # Reject if agent couldn't find information
            rejection_phrases = [
                "no relevant information",
                "maximum iterations reached",
                "max iterations reached",
                "i don't have specific information",
                "i don't have information",
                "could not find",
                "no information found",
                "no results found"
            ]
            
            if any(phrase in result_lower for phrase in rejection_phrases):
                logger.info("Agent search returned insufficient results - will fallback to vector search")
                return None
            
            # Accept the result - it's valid
            logger.info(f"Agent found relevant information (length: {len(result)})")
            return result
            
        except Exception as e:
            logger.error(f"Agent search failed: {e}")
            return None

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Pure vector search using LangChain (NO agent here).
        This is the fallback method called after agent fails.
        """
        try:
            logger.info(f"🔍 Vector search for: '{query}', top_k={top_k}")
            
            # Use LangChain's similarity search with score
            langchain_results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            logger.info(f"📊 LangChain returned {len(langchain_results)} results")
            
            # Convert to your format
            results = []
            for doc, score in langchain_results:
                results.append({
                    'chunk_id': doc.metadata.get('id', 0),
                    'document_id': doc.metadata.get('id', 0),
                    'filename': doc.metadata.get('filename', 'unknown'),
                    'text': doc.page_content,
                    'score': float(1 - score),  # Convert distance to similarity
                    'source': doc.metadata.get('source', 'pdf'),
                    'metadata': {
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'document_metadata': {k: v for k, v in doc.metadata.items() 
                                            if k not in ['filename', 'chunk_index', 'source', 'created_at']},
                        'created_at': doc.metadata.get('created_at', int(time.time())),
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in vector search: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Query LangChain's embedding table
            with self.SessionLocal() as session:
                # Count total chunks
                result = session.execute(
                    text("SELECT COUNT(*) FROM langchain_pg_embedding")
                ).scalar()
                total_chunks = result if result else 0
                
                # Count unique documents by grouping by filename and file_path (same logic as get_all_documents)
                # This matches the GROUP BY in get_all_documents: GROUP BY e.cmetadata->>'filename', e.cmetadata->>'file_path'
                result = session.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM (
                            SELECT DISTINCT 
                                e.cmetadata->>'filename' AS filename,
                                e.cmetadata->>'file_path' AS file_path
                            FROM langchain_pg_embedding e
                            WHERE e.cmetadata->>'filename' IS NOT NULL
                        ) AS unique_docs
                    """)
                ).scalar()
                total_docs = result if result else 0
                
                return {
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                    'index_size': total_chunks,
                    'embedding_dimension': self.dimension,
                    'max_workers': self.max_workers,
                    'batch_size': self.batch_size,
                }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'index_size': 0,
                'embedding_dimension': self.dimension,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
            }

    def clear_index(self):
        """Clear the LangChain vectorstore index."""
        try:
            # Delete all documents from vectorstore
            with self.SessionLocal() as session:
                session.execute(text("TRUNCATE TABLE langchain_pg_embedding"))
                session.execute(text("TRUNCATE TABLE langchain_pg_collection CASCADE"))
                session.commit()
            logger.info("LangChain vectorstore cleared")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from LangChain vectorstore, grouped by filename."""
        try:
            with self.SessionLocal() as session:
                # Query to get unique documents based on filename in cmetadata
                result = session.execute(
                    text("""
                        SELECT 
                            MIN(e.id) AS doc_id,
                            e.cmetadata->>'filename' AS filename,
                            e.cmetadata->>'file_path' AS file_path,
                            COUNT(*) AS total_chunks,
                            (array_agg(e.cmetadata ORDER BY e.id))[1] AS metadata
                        FROM langchain_pg_embedding e
                        WHERE e.cmetadata->>'filename' IS NOT NULL
                        GROUP BY e.cmetadata->>'filename', e.cmetadata->>'file_path'
                        ORDER BY MIN(e.id) ASC
                    """)
                ).fetchall()

                documents: List[Dict[str, Any]] = []
                for row in result:
                    doc_id = row[0]
                    filename = row[1] or 'unknown'
                    file_path = row[2] or ''
                    total_chunks = int(row[3]) if row[3] is not None else 0
                    chunk_metadata = row[4] if row[4] else {}
                    
                    # Build document-level metadata, filtering out chunk-specific fields
                    metadata = {}
                    chunk_specific_fields = {'chunk_index', 'row_number', 'row_data'}
                    
                    # Extract document-level fields from chunk metadata
                    document_level_fields = {
                        'file_size', 'source_type', 'source_url', 'processing_timestamp', 
                        'total_sheets', 'chunk_size', 'chunk_overlap', 'source', 'domain', 
                        'scraped_at', 'content_type', 'headings'
                    }
                    
                    for key, value in chunk_metadata.items():
                        if key not in chunk_specific_fields:
                            metadata[key] = value
                    
                    # Ensure filename and file_path are set
                    metadata['filename'] = filename
                    metadata['file_path'] = file_path
                    
                    # If file_size is missing, try to get it from any chunk or from the file
                    if 'file_size' not in metadata or metadata.get('file_size') is None:
                        # First, try to find file_size in any chunk for this document
                        file_size_result = session.execute(
                            text("""
                                SELECT e.cmetadata->>'file_size' AS file_size
                                FROM langchain_pg_embedding e
                                WHERE e.cmetadata->>'filename' = :filename
                                  AND e.cmetadata->>'file_size' IS NOT NULL
                                LIMIT 1
                            """),
                            {"filename": filename}
                        ).fetchone()
                        
                        if file_size_result and file_size_result[0]:
                            try:
                                metadata['file_size'] = int(file_size_result[0])
                            except (ValueError, TypeError):
                                pass
                        
                        # If still not found, try to get it from the file system
                        if ('file_size' not in metadata or metadata.get('file_size') is None) and file_path and not file_path.startswith('http'):
                            try:
                                from pathlib import Path
                                import os
                                
                                # Try multiple path resolution strategies
                                file_path_obj = None
                                
                                # Strategy 1: Use path as-is (absolute or relative to current working directory)
                                test_path = Path(file_path)
                                if test_path.exists():
                                    file_path_obj = test_path
                                else:
                                    # Strategy 2: Normalize backslashes and try again
                                    normalized_path = file_path.replace('\\', os.sep)
                                    test_path = Path(normalized_path)
                                    if test_path.exists():
                                        file_path_obj = test_path
                                    else:
                                        # Strategy 3: Try relative to project root (if we can determine it)
                                        # Get the directory of this file (vector_store.py) and go up to project root
                                        current_file_dir = Path(__file__).parent.parent
                                        test_path = current_file_dir / normalized_path
                                        if test_path.exists():
                                            file_path_obj = test_path
                                
                                if file_path_obj and file_path_obj.exists():
                                    metadata['file_size'] = file_path_obj.stat().st_size
                                    logger.debug(f"Added file_size {metadata['file_size']} for {filename} from file path: {file_path_obj}")
                            except Exception as e:
                                logger.warning(f"Could not get file_size for {file_path}: {e}")

                    documents.append({
                        'id': doc_id,
                        'filename': filename,
                        'file_path': file_path,
                        'total_chunks': total_chunks,
                        'metadata': metadata,
                    })

                return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    def delete_document(self, filename: str) -> bool:
        """Delete all chunks for a document by filename."""
        try:
            with self.SessionLocal() as session:
                # Delete all chunks where filename matches
                result = session.execute(
                    text("""
                        DELETE FROM langchain_pg_embedding
                        WHERE cmetadata->>'filename' = :filename
                    """),
                    {"filename": filename}
                )
                deleted_count = result.rowcount
                session.commit()
                logger.info(f"Deleted {deleted_count} chunks for document: {filename}")
                return deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return False

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        try:
            with self.SessionLocal() as session:
                result = session.execute(
                    text("""
                        SELECT 
                            e.id,
                            e.document as text,
                            e.cmetadata
                        FROM langchain_pg_embedding e
                        WHERE e.collection_id = :doc_id
                        ORDER BY (e.cmetadata->>'chunk_index')::int
                    """),
                    {"doc_id": doc_id}
                ).fetchall()
                
                return [
                    {
                        'id': row[0],
                        'text': row[1],
                        'metadata': row[2] if row[2] else {}
                    }
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a single document/collection by its id with metadata and chunk count."""
        try:
            with self.SessionLocal() as session:
                row = session.execute(
                    text(
                        """
                        SELECT 
                            c.id AS collection_id,
                            c.name AS collection_name,
                            c.cmetadata AS collection_metadata,
                            COALESCE(cnt.total_chunks, 0) AS total_chunks
                        FROM langchain_pg_collection c
                        LEFT JOIN (
                            SELECT e.collection_id, COUNT(*) AS total_chunks
                            FROM langchain_pg_embedding e
                            GROUP BY e.collection_id
                        ) AS cnt ON cnt.collection_id = c.id
                        WHERE c.id = :doc_id
                        """
                    ),
                    {"doc_id": doc_id}
                ).fetchone()

                if not row:
                    return None

                collection_id = row[0]
                collection_name = row[1]
                collection_metadata = row[2] if row[2] else {}
                total_chunks = int(row[3]) if row[3] is not None else 0

                file_path = ""
                if isinstance(collection_metadata, dict):
                    file_path = collection_metadata.get('file_path', "") or ""

                return {
                    'id': collection_id,
                    'filename': collection_name,
                    'file_path': file_path,
                    'total_chunks': total_chunks,
                    'metadata': collection_metadata,
                }
        except Exception as e:
            logger.error(f"Error getting document by id: {e}")
            return None