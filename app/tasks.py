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


def safe_task_wrapper(func):
    """Wrapper to ensure all exceptions are properly handled and serialized."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            import json
            
            # Get the task instance if it's a bound task
            task_instance = args[0] if args and hasattr(args[0], 'update_state') else None
            
            error_msg = str(e)
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            traceback_str = error_traceback[:1000] if error_traceback else ""
            
            logger.error(f"Task {func.__name__} failed: {error_type}: {error_msg}\n{error_traceback}")
            
            # Create a safe, JSON-serializable error result
            error_result = {
                'success': False,
                'message': f'Task failed: {error_msg}',
                'error': error_msg,
                'error_type': error_type,
                'progress': 100
            }
            
            # Try to update state if we have a task instance
            if task_instance:
                try:
                    safe_meta = {
                        'error': error_msg,
                        'error_type': error_type,
                        'traceback': traceback_str
                    }
                    json.dumps(safe_meta)  # Verify it's serializable
                    task_instance.update_state(state=states.FAILURE, meta=safe_meta)
                except Exception as state_error:
                    logger.error(f"Failed to update task state: {state_error}")
            
            # Return the error result instead of raising
            return error_result
    return wrapper


@shared_task(bind=True, name="app.tasks.ingest_documents_task", acks_late=True, reject_on_worker_lost=True)
def ingest_documents_task(self, file_paths: List[str], max_workers: int = 4, batch_size: int = 100) -> Dict[str, Any]:
    """Background ingestion with progress updates."""
    task_id = self.request.id
    
    # Log task start with clear markers
    logger.info("=" * 100)
    logger.info(f"TASK STARTED: {task_id}")
    logger.info(f"Function: ingest_documents_task")
    logger.info(f"Files to process: {file_paths}")
    logger.info(f"Max workers: {max_workers}, Batch size: {batch_size}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info("=" * 100)
    
    # Import here to ensure fresh imports in worker process
    from app.rag_pipeline import RAGPipeline
    from app.document_processor import DocumentProcessor
    
    try:
        # Update state to show task has started
        try:
            self.update_state(state='PROGRESS', meta={
                'step': 'initializing',
                'progress': 0,
                'message': 'Task started',
                'task_id': task_id
            })
            logger.info(f"Task {task_id}: State updated to PROGRESS (initializing)")
        except Exception as state_error:
            logger.warning(f"Could not update initial state: {state_error}")
        
        # Validate inputs first
        if not file_paths:
            error_msg = "No file paths provided"
            logger.error(f"Task {task_id}: {error_msg}")
            raise ValueError(error_msg)
        
        if not isinstance(file_paths, list):
            error_msg = f"file_paths must be a list, got {type(file_paths)}"
            logger.error(f"Task {task_id}: {error_msg}")
            raise TypeError(error_msg)
        
        # Verify all files exist before processing
        for path_str in file_paths:
            path = Path(path_str)
            if not path.exists():
                error_msg = f"File not found: {path}"
                logger.error(f"Task {task_id}: {error_msg}")
                raise FileNotFoundError(error_msg)
            logger.info(f"Task {task_id}: Verified file exists: {path.name} ({path.stat().st_size} bytes)")
        
        logger.info(f"Task {task_id}: Initializing RAGPipeline and DocumentProcessor")
        
        try:
            pipeline = RAGPipeline()
            processor = DocumentProcessor()
            logger.info(f"Task {task_id}: Initialization complete")
        except Exception as init_error:
            logger.error(f"Task {task_id}: Failed to initialize components: {init_error}")
            raise

        total_files = len(file_paths)
        processed_docs = []
        total_chunks = 0
        failed_files = []

        for idx, path_str in enumerate(file_paths, start=1):
            path = Path(path_str)
            logger.info(f"Task {task_id}: Processing file {idx}/{total_files}: {path.name}")
            
            try:
                self.update_state(state='PROGRESS', meta={
                    'step': 'processing_files',
                    'current_file': path.name,
                    'files_processed': idx - 1,
                    'total_files': total_files,
                    'progress': int(((idx - 1) / max(total_files, 1)) * 25),
                    'task_id': task_id
                })
                logger.info(f"Task {task_id}: Progress updated - {idx-1}/{total_files} files processed")
            except Exception as state_error:
                logger.warning(f"Task {task_id}: Could not update state (non-critical): {state_error}")
            
            try:
                logger.info(f"Task {task_id}: Starting document processing for {path.name}")
                doc_info = processor.process_document(path)
                
                logger.info(f"Task {task_id}: Document processing returned: {doc_info}")
                
                if doc_info and doc_info.get('total_chunks', 0) > 0:
                    processed_docs.append(doc_info)
                    total_chunks += doc_info.get('total_chunks', 0)
                    logger.info(f"Task {task_id}: Successfully processed {path.name}: {doc_info.get('total_chunks', 0)} chunks")
                else:
                    error_msg = f"File {path.name} produced no chunks. doc_info: {doc_info}"
                    logger.warning(f"Task {task_id}: {error_msg}")
                    failed_files.append({'file': path.name, 'error': 'No chunks produced'})
                    
            except Exception as e:
                error_trace = traceback.format_exc()
                error_msg = f"Failed processing {path.name}: {str(e)}"
                logger.error(f"Task {task_id}: {error_msg}\n{error_trace}")
                failed_files.append({'file': path.name, 'error': str(e)})
                continue

        logger.info(f"Task {task_id}: File processing complete. Processed: {len(processed_docs)}, Failed: {len(failed_files)}")
        
        if not processed_docs:
            error_msg = f'No documents processed successfully. Total files: {total_files}, Failed: {len(failed_files)}'
            if failed_files:
                error_msg += f'\nFailed files: {failed_files}'
            logger.error(f"Task {task_id}: {error_msg}")
            
            return {
                'success': False, 
                'message': error_msg,
                'failed_files': failed_files,
                'total_files': total_files,
                'progress': 100,
                'task_id': task_id
            }

        # Configure store
        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        try:
            self.update_state(state='PROGRESS', meta={
                'step': 'indexing',
                'processed_documents': len(processed_docs),
                'total_chunks': total_chunks,
                'progress': 40,
                'task_id': task_id
            })
            logger.info(f"Task {task_id}: State updated - starting indexing")
        except Exception as state_error:
            logger.warning(f"Task {task_id}: Could not update state (non-critical): {state_error}")

        logger.info(f"Task {task_id}: Starting vector store indexing for {len(processed_docs)} documents, {total_chunks} chunks")
        
        try:
            stats = pipeline.vector_store.add_documents_parallel(processed_docs)
            logger.info(f"Task {task_id}: Vector store indexing completed: {stats}")
        except Exception as index_error:
            error_trace = traceback.format_exc()
            logger.error(f"Task {task_id}: Vector store indexing failed: {index_error}\n{error_trace}")
            raise

        try:
            self.update_state(state='PROGRESS', meta={
                'step': 'finalizing',
                'progress': 90,
                'task_id': task_id
            })
            logger.info(f"Task {task_id}: State updated - finalizing")
        except Exception as state_error:
            logger.warning(f"Task {task_id}: Could not update state (non-critical): {state_error}")

        success = bool(stats.get('success'))
        result = {
            'success': success,
            'message': 'Ingestion finished' if success else 'Ingestion completed with errors',
            'documents_processed': stats.get('total_documents', len(processed_docs)),
            'chunks_created': stats.get('processed_chunks', total_chunks),
            'processing_time': stats.get('processing_time'),
            'chunks_per_second': stats.get('chunks_per_second'),
            'successful_batches': stats.get('successful_batches'),
            'failed_batches': stats.get('failed_batches'),
            'total_batches': stats.get('total_batches'),
            'failed_files': failed_files if failed_files else [],
            'progress': 100,
            'task_id': task_id
        }
        
        logger.info("=" * 100)
        logger.info(f"TASK COMPLETED SUCCESSFULLY: {task_id}")
        logger.info(f"Result: {result}")
        logger.info("=" * 100)
        
        return result
        
    except Exception as e:
        import json
        
        # Ensure all error information is JSON-serializable
        error_msg = str(e)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        
        # Limit traceback size and ensure it's a string
        traceback_str = error_traceback[:1000] if error_traceback else ""
        
        logger.error("=" * 100)
        logger.error(f"TASK FAILED: {task_id}")
        logger.error(f"Error Type: {error_type}")
        logger.error(f"Error Message: {error_msg}")
        logger.error(f"Full Traceback:\n{error_traceback}")
        logger.error("=" * 100)
        
        # Create a safe, JSON-serializable error result
        error_result = {
            'success': False,
            'message': f'Ingestion failed: {error_msg}',
            'error': error_msg,
            'error_type': error_type,
            'progress': 100,
            'task_id': task_id
        }
        
        # Try to update state, but don't fail if it can't be stored
        try:
            safe_meta = {
                'error': error_msg,
                'error_type': error_type,
                'traceback': traceback_str,
                'task_id': task_id
            }
            json.dumps(safe_meta)  # Verify it's serializable
            self.update_state(state=states.FAILURE, meta=safe_meta)
            logger.info(f"Task {task_id}: Error state updated")
        except (TypeError, ValueError, Exception) as state_error:
            logger.error(f"Task {task_id}: Failed to update task state (non-critical): {state_error}")
        
        return error_result


@shared_task(bind=True, name="app.tasks.weblink_ingestion_task", acks_late=True, reject_on_worker_lost=True)
def weblink_ingestion_task(self, url: str, depth: int = 1, max_workers: int = 4, batch_size: int = 100, use_proxy: bool = False, proxy_list: List[str] = None) -> Dict[str, Any]:
    """Scrape a URL (optionally following internal links), process content, and index in background."""
    task_id = self.request.id
    
    logger.info("=" * 100)
    logger.info(f"WEBLINK TASK STARTED: {task_id}")
    logger.info(f"URL: {url}, Depth: {depth}, Use Proxy: {use_proxy}")
    logger.info("=" * 100)
    
    # Import here to ensure fresh imports
    from app.api import scrape_url
    from app.rag_pipeline import RAGPipeline
    from app.document_processor import DocumentProcessor
    
    try:
        # Update initial state
        try:
            self.update_state(state='PROGRESS', meta={
                'step': 'initializing',
                'progress': 0,
                'message': 'Task started',
                'task_id': task_id
            })
            logger.info(f"Task {task_id}: State updated to PROGRESS (initializing)")
        except Exception as state_error:
            logger.warning(f"Could not update initial state: {state_error}")

        pipeline = RAGPipeline()
        processor = DocumentProcessor()
        
        logger.info(f"Task {task_id}: Initialization complete")

        self.update_state(state='PROGRESS', meta={
            'step': 'scraping',
            'target_url': url,
            'progress': 5,
            'task_id': task_id
        })
        logger.info(f"Task {task_id}: Starting web scraping")

        # Handle proxy_list properly
        if proxy_list is None:
            proxy_list = []
        
        scraped_data = scrape_url(url, depth, use_proxy=use_proxy, proxy_list=proxy_list)
        total_items = len(scraped_data)
        
        logger.info(f"Task {task_id}: Scraped {total_items} pages")

        if total_items == 0:
            logger.warning(f"Task {task_id}: No data scraped from {url}")
            return {
                'success': False,
                'message': 'No data scraped',
                'progress': 100,
                'task_id': task_id
            }

        processed_docs = []
        total_content_length = 0

        for idx, item in enumerate(scraped_data, start=1):
            try:
                self.update_state(state='PROGRESS', meta={
                    'step': 'processing_content',
                    'current_url': item.get('url'),
                    'items_processed': idx - 1,
                    'total_items': total_items,
                    'progress': 5 + int(((idx - 1) / max(total_items, 1)) * 35),
                    'task_id': task_id
                })
                logger.info(f"Task {task_id}: Processing item {idx}/{total_items}")
            except Exception as state_error:
                logger.warning(f"Could not update state: {state_error}")

            if 'error' in item:
                logger.warning(f"Task {task_id}: Skipping item with error: {item.get('error')}")
                continue

            full_text = ''
            headings = item.get('headings', []) or []
            if headings:
                full_text += " ".join(headings) + "\n\n"
            if item.get('text'):
                full_text += item['text']

            if not full_text.strip():
                logger.warning(f"Task {task_id}: Skipping item with no content")
                continue

            total_content_length += len(full_text)
            doc_info = processor.process_web_content(url=item.get('url', url), content=full_text, headings=headings)
            if doc_info:
                processed_docs.append(doc_info)
                logger.info(f"Task {task_id}: Processed content from {item.get('url', url)}")

        if not processed_docs:
            logger.warning(f"Task {task_id}: No content processed from scraped pages")
            return {
                'success': False,
                'message': 'No content processed from scraped pages',
                'progress': 100,
                'task_id': task_id
            }

        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        self.update_state(state='PROGRESS', meta={
            'step': 'indexing',
            'processed_documents': len(processed_docs),
            'progress': 60,
            'task_id': task_id
        })
        logger.info(f"Task {task_id}: Starting vector store indexing")

        stats = pipeline.vector_store.add_documents_parallel(processed_docs)
        logger.info(f"Task {task_id}: Indexing completed: {stats}")

        self.update_state(state='PROGRESS', meta={
            'step': 'finalizing',
            'progress': 90,
            'task_id': task_id
        })

        success = bool(stats.get('success'))
        result = {
            'success': success,
            'message': 'Weblink ingestion finished' if success else 'Weblink ingestion completed with errors',
            'urls_processed': len(processed_docs),
            'chunks_created': stats.get('processed_chunks'),
            'processing_time': stats.get('processing_time'),
            'chunks_per_second': stats.get('chunks_per_second'),
            'total_content_length': total_content_length,
            'successful_batches': stats.get('successful_batches'),
            'failed_batches': stats.get('failed_batches'),
            'total_batches': stats.get('total_batches'),
            'progress': 100,
            'task_id': task_id
        }
        
        logger.info("=" * 100)
        logger.info(f"WEBLINK TASK COMPLETED: {task_id}")
        logger.info(f"Result: {result}")
        logger.info("=" * 100)
        
        return result
        
    except Exception as e:
        import json
        
        error_msg = str(e)
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        traceback_str = error_traceback[:1000] if error_traceback else ""
        
        logger.error("=" * 100)
        logger.error(f"WEBLINK TASK FAILED: {task_id}")
        logger.error(f"Error: {error_msg}")
        logger.error(f"Traceback:\n{error_traceback}")
        logger.error("=" * 100)
        
        error_result = {
            'success': False,
            'message': f'Weblink ingestion failed: {error_msg}',
            'error': error_msg,
            'error_type': error_type,
            'progress': 100,
            'task_id': task_id
        }
        
        try:
            safe_meta = {
                'error': error_msg,
                'error_type': error_type,
                'traceback': traceback_str,
                'task_id': task_id
            }
            json.dumps(safe_meta)
            self.update_state(state=states.FAILURE, meta=safe_meta)
        except (TypeError, ValueError, Exception) as state_error:
            logger.error(f"Failed to update task state: {state_error}")
        
        return error_result