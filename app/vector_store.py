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
    # Optional: pgvector backend
    from sqlalchemy import Column, Integer, String, Text, JSON, create_engine, select, func
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
    """Enhanced PostgreSQL + pgvector-based vector store with parallel processing."""
    
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
            pool_size=max_workers * 2,  # More connections for parallel processing
            max_overflow=max_workers,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self._init_pg_base()
        self._setup_database()
    

    def _init_pg_base(self):
        global Base
        if Base is None:
            from sqlalchemy.orm import declarative_base as _declarative_base
            Base = _declarative_base()
        
        class EmbeddingData(Base):
            __tablename__ = "embedding_data"
            __table_args__ = {"extend_existing": True}

            id = Column(Integer, primary_key=True, autoincrement=True)
            filename = Column(String(512), nullable=False)
            file_path = Column(Text, nullable=False)
            chunk_index = Column(Integer, nullable=False)
            text = Column(Text, nullable=False)
            embedding = Column(Vector(self.dimension))
            document_metadata = Column(JSON)
            total_chunks = Column(Integer, nullable=False)

        self.EmbeddingData = EmbeddingData

    def _setup_database(self):
        """Setup database with vector extension and tables."""
        try:
            with self.SessionLocal() as session:
                session.execute(func.create_extension('vector', if_not_exists=True))
                session.commit()
                logger.info("Vector extension enabled")
        except Exception as e:
            logger.warning(f"Could not enable vector extension: {e}")
        
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def _create_chunk_batches(self, documents: List[Dict[str, Any]]) -> List[ChunkBatch]:
        """Create batches of chunks for parallel processing."""
        batches = []
        batch_id = 0
        
        for doc in documents:
            chunks = doc['chunks']
            total_chunks = len(chunks)
            
            # Create batches for this document
            for i in range(0, total_chunks, self.batch_size):
                batch_chunks = []
                batch_end = min(i + self.batch_size, total_chunks)
                
                for chunk_idx in range(i, batch_end):
                    chunk_data = {
                        'filename': doc['filename'],
                        'file_path': doc['file_path'],
                        'chunk_index': chunk_idx,
                        'text': chunks[chunk_idx],
                        'document_metadata': doc.get('metadata', {}),
                        'total_chunks': doc['total_chunks']
                    }
                    batch_chunks.append(chunk_data)
                
                batches.append(ChunkBatch(
                    batch_id=batch_id,
                    chunks=batch_chunks,
                    document_info={
                        'filename': doc['filename'],
                        'file_path': doc['file_path'],
                        'total_chunks': doc['total_chunks']
                    }
                ))
                batch_id += 1
        
        return batches

    def _process_chunk_batch(self, batch: ChunkBatch) -> Dict[str, Any]:
        """Process a single batch of chunks."""
        session = None
        try:
            # Create new session for this worker
            session = self.SessionLocal()
            
            # Generate embeddings for all chunks in batch
            texts = [chunk['text'] for chunk in batch.chunks]
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Create embedding data objects
            embedding_rows = []
            for i, chunk in enumerate(batch.chunks):
                embedding_rows.append(
                    self.EmbeddingData(
                        filename=chunk['filename'],
                        file_path=chunk['file_path'],
                        chunk_index=chunk['chunk_index'],
                        text=chunk['text'],
                        embedding=embeddings[i].tolist(),
                        document_metadata=chunk['document_metadata'],
                        total_chunks=chunk['total_chunks']
                    )
                )
            
            # Bulk insert
            session.add_all(embedding_rows)
            session.commit()
            
            # Update progress
            with self.progress_lock:
                self.progress.processed_chunks += len(batch.chunks)
                self.progress.current_batch += 1
                
                elapsed = time.time() - self.progress.start_time
                chunks_per_second = self.progress.processed_chunks / elapsed if elapsed > 0 else 0
                eta = (self.progress.total_chunks - self.progress.processed_chunks) / chunks_per_second if chunks_per_second > 0 else 0
                
                logger.info(
                    f"Batch {batch.batch_id} completed: {self.progress.processed_chunks}/{self.progress.total_chunks} chunks "
                    f"({chunks_per_second:.1f} chunks/sec, ETA: {eta:.1f}s)"
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
            
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            return {
                'batch_id': batch.batch_id,
                'status': 'error',
                'error': str(e),
                'chunks_failed': len(batch.chunks),
                'filename': batch.document_info['filename']
            }
        finally:
            if session:
                session.close()

    def add_documents_parallel(self, documents: List[Dict[str, Any]], 
                             progress_callback=None) -> Dict[str, Any]:
        """Add documents using parallel processing."""
        start_time = time.time()
        
        # Initialize progress tracking
        total_chunks = sum(doc['total_chunks'] for doc in documents)
        self.progress = IngestionProgress(
            total_chunks=total_chunks,
            start_time=start_time
        )
        
        # Create batches
        batches = self._create_chunk_batches(documents)
        self.progress.total_batches = len(batches)
        
        logger.info(f"Starting parallel ingestion: {len(documents)} documents, "
                   f"{total_chunks} chunks, {len(batches)} batches, {self.max_workers} workers")
        
        # Process batches in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_chunk_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Call progress callback if provided
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
        
        # Calculate final statistics
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

    # Keep original method for backward compatibility
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents (uses parallel processing internally)."""
        result = self.add_documents_parallel(documents)
        return result['success']

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search: semantic vector search + per-word keyword search with IDF weighting."""
        import re, math

        try:
            results = {}

            # --------- 1️⃣ Semantic search (vector search) ---------
            query_vec = self.embedding_model.encode([query], convert_to_tensor=False)[0].tolist()
            with self.SessionLocal() as session:
                semantic_stmt = (
                    select(
                        self.EmbeddingData,
                        (1 - self.EmbeddingData.embedding.cosine_distance(query_vec)).label('similarity_score')
                    )
                    .order_by(self.EmbeddingData.embedding.cosine_distance(query_vec))
                    .limit(top_k)
                )
                semantic_rows = session.execute(semantic_stmt).all()
                for row in semantic_rows:
                    embedding_data = row[0]
                    similarity_score = float(row[1]) if row[1] is not None else 0.0
                    results[embedding_data.id] = {
                        'chunk_id': embedding_data.id,
                        'document_id': embedding_data.id,
                        'filename': embedding_data.filename,
                        'text': embedding_data.text,
                        'score': similarity_score,
                        'metadata': {
                            'chunk_index': embedding_data.chunk_index,
                            'document_metadata': embedding_data.document_metadata,
                        }
                    }

            # --------- 2️⃣ Per-word keyword search (TF-IDF style weighting) ---------
            query_tokens = re.findall(r'\w+', query.lower())

            # Expanded stopword list + min length filter
            stopwords = set([
                'the','of','and','in','on','for','a','an','with','to','is',
                'what','who','when','where','which','how','why','this','that',
                'by','as','at','be','or','from','it','are','was','were','can'
            ])
            keywords = [t for t in query_tokens if t not in stopwords and len(t) >= 3]

            with self.SessionLocal() as session:
                # Count total chunks (for IDF)
                total_chunks = session.query(self.EmbeddingData).count()

                keyword_counts = {}
                for token in keywords:
                    stmt = select(self.EmbeddingData).where(self.EmbeddingData.text.ilike(f"%{token}%"))
                    rows = session.execute(stmt).scalars().all()
                    n_chunks_with_token = len(rows)

                    # Compute IDF weight (rare tokens get higher score)
                    idf = math.log((total_chunks + 1) / (1 + n_chunks_with_token))

                    for row in rows:
                        if row.id not in keyword_counts:
                            keyword_counts[row.id] = {'score': 0.0, 'data': row}
                        keyword_counts[row.id]['score'] += idf  # weight instead of +1

                # Merge results
                for item in keyword_counts.values():
                    row = item['data']
                    score = float(item['score'])
                    if row.id in results:
                        results[row.id]['score'] = max(results[row.id]['score'], score)
                    else:
                        results[row.id] = {
                            'chunk_id': row.id,
                            'document_id': row.id,
                            'filename': row.filename,
                            'text': row.text,
                            'score': score,
                            'metadata': {
                                'chunk_index': row.chunk_index,
                                'document_metadata': row.document_metadata,
                            }
                        }

            # --------- 3️⃣ Sort combined results by score ---------
            sorted_results = sorted(results.values(), key=lambda x: x['score'], reverse=True)
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise



    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics with parallel processing info."""
        with self.SessionLocal() as session:
            total_chunks = session.execute(select(func.count(self.EmbeddingData.id))).scalar()
            total_docs = session.execute(
                select(func.count(func.distinct(self.EmbeddingData.filename)))
            ).scalar()
            return {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'index_size': total_chunks,
                'embedding_dimension': self.dimension,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
            }

    def clear_index(self):
        """Clear the index."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        logger.info("PGVector index cleared")

    # Add other methods from original implementation...
    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        with self.SessionLocal() as session:
            stmt = select(self.EmbeddingData).where(self.EmbeddingData.id == doc_id).limit(1)
            embedding_data = session.execute(stmt).scalar_one_or_none()
            if not embedding_data:
                return None
            return {
                'id': embedding_data.id,
                'filename': embedding_data.filename,
                'file_path': embedding_data.file_path,
                'total_chunks': embedding_data.total_chunks,
                'metadata': embedding_data.document_metadata,
            }

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self.SessionLocal() as session:
            embedding_data = session.get(self.EmbeddingData, chunk_id)
            if not embedding_data:
                return None
            return {
                'id': embedding_data.id,
                'document_id': embedding_data.id,
                'chunk_index': embedding_data.chunk_index,
                'text': embedding_data.text,
                'filename': embedding_data.filename,
            }

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            stmt = (
                select(
                    self.EmbeddingData.id,
                    self.EmbeddingData.filename,
                    self.EmbeddingData.file_path,
                    self.EmbeddingData.total_chunks,
                    self.EmbeddingData.document_metadata,
                )
                .distinct(
                    self.EmbeddingData.filename,
                    self.EmbeddingData.file_path,
                    self.EmbeddingData.total_chunks,
                )
                .order_by(
                    self.EmbeddingData.filename,
                    self.EmbeddingData.file_path,
                    self.EmbeddingData.total_chunks,
                    self.EmbeddingData.id,
                )
            )
            rows = session.execute(stmt).all()
            return [
                {
                    'id': row.id,
                    'filename': row.filename,
                    'file_path': row.file_path,
                    'total_chunks': row.total_chunks,
                    'metadata': row.document_metadata,
                }
                for row in rows
            ]

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            doc_stmt = select(self.EmbeddingData.filename).where(self.EmbeddingData.id == doc_id).limit(1)
            filename = session.execute(doc_stmt).scalar_one_or_none()
            if not filename:
                return []
            
            stmt = select(self.EmbeddingData).where(self.EmbeddingData.filename == filename).order_by(self.EmbeddingData.chunk_index)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    'id': c.id,
                    'document_id': c.id,
                    'chunk_index': c.chunk_index,
                    'text': c.text,
                    'filename': c.filename,
                }
                for c in rows
            ]