from typing import List, Dict, Any
from celery import states
from celery import shared_task, current_task
from celery.exceptions import Retry
from loguru import logger
from pathlib import Path
import time
import re
import functools
import sys
import traceback


# def safe_task_wrapper(func):
#     """Wrapper to ensure all exceptions are properly handled and serialized."""
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             import traceback
#             import json
            
#             # Get the task instance if it's a bound task
#             task_instance = args[0] if args and hasattr(args[0], 'update_state') else None
            
#             error_msg = str(e)
#             error_type = type(e).__name__
#             error_traceback = traceback.format_exc()
#             traceback_str = error_traceback[:1000] if error_traceback else ""
            
#             logger.error(f"Task {func.__name__} failed: {error_type}: {error_msg}\n{error_traceback}")
            
#             # Create a safe, JSON-serializable error result
#             error_result = {
#                 'success': False,
#                 'message': f'Task failed: {error_msg}',
#                 'error': error_msg,
#                 'error_type': error_type,
#                 'progress': 100
#             }
            
#             # Try to update state if we have a task instance
#             if task_instance:
#                 try:
#                     safe_meta = {
#                         'error': error_msg,
#                         'error_type': error_type,
#                         'traceback': traceback_str
#                     }
#                     json.dumps(safe_meta)  # Verify it's serializable
#                     task_instance.update_state(state=states.FAILURE, meta=safe_meta)
#                 except Exception as state_error:
#                     logger.error(f"Failed to update task state: {state_error}")
            
#             # Return the error result instead of raising
#             return error_result
#     return wrapper


@shared_task(
    bind=True, 
    name="app.tasks.ingest_documents_task", 
    acks_late=True, 
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 2},
    retry_backoff=True
)
def ingest_documents_task(self, file_paths: List[str], max_workers: int = 4, batch_size: int = 100) -> Dict[str, Any]:
    """Background ingestion with progress updates."""
    task_id = self.request.id
    
    logger.info("=" * 100)
    logger.info(f"TASK STARTED: {task_id}")
    logger.info(f"Files to process: {file_paths}")
    logger.info("=" * 100)
    
    # Import here to ensure fresh imports in worker process
    from app.rag_pipeline import RAGPipeline
    from app.document_processor import DocumentProcessor
    
    pipeline = None
    processor = None
    
    try:
        # Update state to show task has started
        self.update_state(state='PROGRESS', meta={
            'step': 'initializing',
            'progress': 0,
            'message': 'Task started',
            'task_id': task_id
        })
        
        # Validate inputs first
        if not file_paths:
            raise ValueError("No file paths provided")
        
        if not isinstance(file_paths, list):
            raise TypeError(f"file_paths must be a list, got {type(file_paths)}")
        
        # Verify all files exist before processing
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            logger.info(f"Task {task_id}: Verified file exists: {path.name}")
        
        logger.info(f"Task {task_id}: Initializing RAGPipeline and DocumentProcessor")
        
        pipeline = RAGPipeline()
        processor = DocumentProcessor()
        logger.info(f"Task {task_id}: Initialization complete")

        total_files = len(file_paths)
        processed_docs = []
        total_chunks = 0
        failed_files = []

        for idx, path_str in enumerate(file_paths, start=1):
            path = Path(path_str)
            logger.info(f"Task {task_id}: Processing file {idx}/{total_files}: {path.name}")
            
            self.update_state(state='PROGRESS', meta={
                'step': 'processing_files',
                'current_file': path.name,
                'files_processed': idx - 1,
                'total_files': total_files,
                'progress': int(((idx - 1) / max(total_files, 1)) * 25),
                'task_id': task_id
            })
            
            try:
                logger.info(f"Task {task_id}: Starting document processing for {path.name}")
                doc_info = processor.process_document(path)
                
                if doc_info and doc_info.get('total_chunks', 0) > 0:
                    processed_docs.append(doc_info)
                    total_chunks += doc_info.get('total_chunks', 0)
                    logger.info(f"Task {task_id}: Successfully processed {path.name}")
                else:
                    logger.warning(f"Task {task_id}: File {path.name} produced no chunks")
                    failed_files.append({'file': path.name, 'error': 'No chunks produced'})
                    
            except Exception as e:
                error_msg = f"Failed processing {path.name}: {str(e)}"
                logger.error(f"Task {task_id}: {error_msg}")
                failed_files.append({'file': path.name, 'error': str(e)})
                continue

        logger.info(f"Task {task_id}: File processing complete. Processed: {len(processed_docs)}")
        
        if not processed_docs:
            raise ValueError(f'No documents processed successfully. Failed: {len(failed_files)}')

        # Configure store
        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        self.update_state(state='PROGRESS', meta={
            'step': 'indexing',
            'processed_documents': len(processed_docs),
            'total_chunks': total_chunks,
            'progress': 40,
            'task_id': task_id
        })

        logger.info(f"Task {task_id}: Starting vector store indexing")
        stats = pipeline.vector_store.add_documents_parallel(processed_docs)
        logger.info(f"Task {task_id}: Vector store indexing completed")

        self.update_state(state='PROGRESS', meta={
            'step': 'finalizing',
            'progress': 90,
            'task_id': task_id
        })

        result = {
            'success': True,
            'message': 'Ingestion finished',
            'documents_processed': stats.get('total_documents', len(processed_docs)),
            'chunks_created': stats.get('processed_chunks', total_chunks),
            'processing_time': stats.get('processing_time'),
            'failed_files': failed_files,
            'progress': 100,
            'task_id': task_id
        }
        
        logger.info(f"TASK COMPLETED SUCCESSFULLY: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"TASK FAILED: {task_id} - {str(e)}")
        logger.error(traceback.format_exc())
        # RE-RAISE the exception instead of returning error dict
        raise
        
    finally:
        # CLEANUP RESOURCES
        if pipeline:
            try:
                if hasattr(pipeline.vector_store, 'close'):
                    pipeline.vector_store.close()
                elif hasattr(pipeline.vector_store, 'conn'):
                    pipeline.vector_store.conn.close()
            except Exception as e:
                logger.warning(f"Error closing pipeline resources: {e}")
        
        if processor:
            try:
                # Close any open file handles if needed
                pass
            except Exception as e:
                logger.warning(f"Error closing processor resources: {e}")
        
        logger.info(f"Task {task_id}: Resources cleaned up")


@shared_task(
    bind=True, 
    name="app.tasks.weblink_ingestion_task", 
    acks_late=True, 
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 2},
    retry_backoff=True
)
def weblink_ingestion_task(self, url: str, depth: int = 1, max_workers: int = 4, batch_size: int = 100, use_proxy: bool = False, proxy_list: List[str] = None) -> Dict[str, Any]:
    """Scrape a URL and index in background."""
    task_id = self.request.id
    
    logger.info("=" * 100)
    logger.info(f"WEBLINK TASK STARTED: {task_id}")
    logger.info(f"URL: {url}, Depth: {depth}")
    logger.info("=" * 100)
    
    from app.api import scrape_url
    from app.rag_pipeline import RAGPipeline
    from app.document_processor import DocumentProcessor
    
    pipeline = None
    processor = None
    
    try:
        self.update_state(state='PROGRESS', meta={
            'step': 'initializing',
            'progress': 0,
            'message': 'Task started',
            'task_id': task_id
        })

        pipeline = RAGPipeline()
        processor = DocumentProcessor()
        
        logger.info(f"Task {task_id}: Initialization complete")

        self.update_state(state='PROGRESS', meta={
            'step': 'scraping',
            'target_url': url,
            'progress': 5,
            'task_id': task_id
        })

        if proxy_list is None:
            proxy_list = []
        
        scraped_data = scrape_url(url, depth, use_proxy=use_proxy, proxy_list=proxy_list)
        total_items = len(scraped_data)
        
        logger.info(f"Task {task_id}: Scraped {total_items} pages")

        if total_items == 0:
            raise ValueError('No data scraped from URL')

        processed_docs = []
        total_content_length = 0

        for idx, item in enumerate(scraped_data, start=1):
            self.update_state(state='PROGRESS', meta={
                'step': 'processing_content',
                'current_url': item.get('url'),
                'items_processed': idx - 1,
                'total_items': total_items,
                'progress': 5 + int(((idx - 1) / max(total_items, 1)) * 35),
                'task_id': task_id
            })

            if 'error' in item:
                logger.warning(f"Task {task_id}: Skipping item with error")
                continue

            full_text = ''
            headings = item.get('headings', []) or []
            if headings:
                full_text += " ".join(headings) + "\n\n"
            if item.get('text'):
                full_text += item['text']

            if not full_text.strip():
                continue

            total_content_length += len(full_text)
            doc_info = processor.process_web_content(url=item.get('url', url), content=full_text, headings=headings)
            if doc_info:
                processed_docs.append(doc_info)

        if not processed_docs:
            raise ValueError('No content processed from scraped pages')

        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        self.update_state(state='PROGRESS', meta={
            'step': 'indexing',
            'processed_documents': len(processed_docs),
            'progress': 60,
            'task_id': task_id
        })

        stats = pipeline.vector_store.add_documents_parallel(processed_docs)
        logger.info(f"Task {task_id}: Indexing completed")

        self.update_state(state='PROGRESS', meta={
            'step': 'finalizing',
            'progress': 90,
            'task_id': task_id
        })

        result = {
            'success': True,
            'message': 'Weblink ingestion finished',
            'urls_processed': len(processed_docs),
            'chunks_created': stats.get('processed_chunks'),
            'processing_time': stats.get('processing_time'),
            'progress': 100,
            'task_id': task_id
        }
        
        logger.info(f"WEBLINK TASK COMPLETED: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"WEBLINK TASK FAILED: {task_id} - {str(e)}")
        logger.error(traceback.format_exc())
        # RE-RAISE the exception
        raise
        
    finally:
        # CLEANUP RESOURCES
        if pipeline:
            try:
                if hasattr(pipeline.vector_store, 'close'):
                    pipeline.vector_store.close()
                elif hasattr(pipeline.vector_store, 'conn'):
                    pipeline.vector_store.conn.close()
            except Exception as e:
                logger.warning(f"Error closing pipeline resources: {e}")
        
        if processor:
            try:
                pass  # Add cleanup if needed
            except Exception as e:
                logger.warning(f"Error closing processor resources: {e}")
        
        logger.info(f"Task {task_id}: Resources cleaned up")