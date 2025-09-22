from typing import List, Dict, Any
from celery import states
from celery import shared_task, current_task
from loguru import logger
from pathlib import Path
import time

from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor


@shared_task(bind=True, name="app.tasks.ingest_documents_task")
def ingest_documents_task(self, file_paths: List[str], max_workers: int = 4, batch_size: int = 100) -> Dict[str, Any]:
    """Background ingestion with progress updates."""
    try:
        pipeline = RAGPipeline()
        processor = DocumentProcessor()

        total_files = len(file_paths)
        processed_docs = []
        total_chunks = 0

        for idx, path_str in enumerate(file_paths, start=1):
            path = Path(path_str)
            self.update_state(state='PROGRESS', meta={
                'step': 'processing_files',
                'current_file': path.name,
                'files_processed': idx - 1,
                'total_files': total_files,
                'progress': int(((idx - 1) / max(total_files, 1)) * 25)
            })
            try:
                doc_info = processor.process_document(path)
                processed_docs.append(doc_info)
                total_chunks += doc_info.get('total_chunks', 0)
            except Exception as e:
                logger.error(f"Failed processing {path.name}: {e}")
                continue

        if not processed_docs:
            return {'success': False, 'message': 'No documents processed', 'progress': 100}

        # Configure store
        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        self.update_state(state='PROGRESS', meta={
            'step': 'indexing',
            'processed_documents': len(processed_docs),
            'total_chunks': total_chunks,
            'progress': 40
        })

        stats = pipeline.vector_store.add_documents_parallel(processed_docs)

        self.update_state(state='PROGRESS', meta={
            'step': 'finalizing',
            'progress': 90
        })

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
            'progress': 100
        }
        return result
    except Exception as e:
        logger.error(f"Ingestion task failed: {e}")
        self.update_state(state=states.FAILURE, meta={'error': str(e)})
        raise


@shared_task(bind=True, name="app.tasks.weblink_ingestion_task")
def weblink_ingestion_task(self, url: str, depth: int = 1, max_workers: int = 4, batch_size: int = 100) -> Dict[str, Any]:
    """Scrape a URL (optionally following internal links), process content, and index in background."""
    try:
        from app.api import scrape_url  # reuse existing scraper

        pipeline = RAGPipeline()
        processor = DocumentProcessor()

        self.update_state(state='PROGRESS', meta={
            'step': 'scraping',
            'target_url': url,
            'progress': 5
        })

        scraped_data = scrape_url(url, depth)
        total_items = len(scraped_data)

        if total_items == 0:
            return {'success': False, 'message': 'No data scraped', 'progress': 100}

        processed_docs = []
        total_content_length = 0

        for idx, item in enumerate(scraped_data, start=1):
            self.update_state(state='PROGRESS', meta={
                'step': 'processing_content',
                'current_url': item.get('url'),
                'items_processed': idx - 1,
                'total_items': total_items,
                'progress': 5 + int(((idx - 1) / max(total_items, 1)) * 35)
            })

            if 'error' in item:
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
            return {'success': False, 'message': 'No content processed from scraped pages', 'progress': 100}

        pipeline.vector_store.max_workers = max_workers
        pipeline.vector_store.batch_size = batch_size

        self.update_state(state='PROGRESS', meta={
            'step': 'indexing',
            'processed_documents': len(processed_docs),
            'progress': 60
        })

        stats = pipeline.vector_store.add_documents_parallel(processed_docs)

        self.update_state(state='PROGRESS', meta={
            'step': 'finalizing',
            'progress': 90
        })

        success = bool(stats.get('success'))
        return {
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
            'progress': 100
        }
    except Exception as e:
        logger.error(f"Weblink task failed: {e}")
        self.update_state(state=states.FAILURE, meta={'error': str(e)})
        raise