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
    from langchain.schema import Document as LangChainDocument
    
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
            role = Column(String(16), nullable=False)
            message = Column(Text, nullable=False)
            embedding = Column(Vector(self.dimension))
            created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))

        self.ChatMessageORM = ChatMessageORM

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

    def save_chat_message(self, session_id: str, role: str, message: str) -> Dict[str, Any]:
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
                "role": obj.role,
                "message": obj.message,
                "created_at": obj.created_at
            }

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetch chat messages for a session ordered by created_at ascending."""
        with self.SessionLocal() as db:
            rows = db.query(self.ChatMessageORM).filter_by(session_id=session_id).order_by(
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

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search using LangChain's similarity_search_with_score method.
        """ 
        try:
            logger.info(f"Searching with LangChain for: '{query}', top_k={top_k}")
            
            # Use LangChain's similarity search with score
            langchain_results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # logger.info(f"LangChain returned {langchain_results} results")
            # print('langchain_results =========================================== ',langchain_results)
            
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
            
            print('langchain_results =========================================== ',results)
            return results
            
        except Exception as e:
            logger.error(f"Error in LangChain similarity search: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Query LangChain's embedding table
            with self.SessionLocal() as session:
                # LangChain stores data in langchain_pg_embedding table
                result = session.execute(
                    text("SELECT COUNT(*) FROM langchain_pg_embedding")
                ).scalar()
                total_chunks = result if result else 0
                
                # Count unique collections (documents)
                result = session.execute(
                    text("SELECT COUNT(*) FROM langchain_pg_collection")
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
                    metadata = row[4] if row[4] else {}

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