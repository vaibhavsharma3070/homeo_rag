from celery import Celery
from celery.signals import task_failure, task_postrun, worker_ready, task_prerun
from loguru import logger
import traceback
import json
from app.config import settings
from app import tasks


celery_app = Celery(
    'homeo_rag',
    broker=getattr(settings, 'celery_broker_url', 'redis://127.0.0.1:6379/0'),
    backend=getattr(settings, 'celery_result_backend', 'redis://127.0.0.1:6379/0'),
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Simplified connection settings
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    
    # Task time limits
    task_time_limit=3600,
    task_soft_time_limit=3300,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Log when worker is ready"""
    logger.info("=" * 100)
    logger.info("CELERY WORKER IS READY AND LISTENING FOR TASKS")
    logger.info(f"Worker: {sender.hostname}")
    logger.info("=" * 100)


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """Log before task execution"""
    logger.info("=" * 100)
    logger.info(f"TASK PRERUN: {task.name}")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Args: {args}")
    logger.info(f"Kwargs: {kwargs}")
    logger.info("=" * 100)


# @task_failure.connect
# def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
#     """Handle task failures and ensure exceptions are properly serialized."""
#     try:
#         # Extract error information safely
#         try:
#             error_msg = str(exception) if exception else "Unknown error"
#             error_type = type(exception).__name__ if exception else "Exception"
#         except Exception:
#             error_msg = "Error converting exception to string"
#             error_type = "Exception"
        
#         # Log the failure with full details
#         logger.error("=" * 100)
#         logger.error(f"TASK FAILURE HANDLER - Task {task_id} failed")
#         logger.error(f"Error Type: {error_type}")
#         logger.error(f"Error Message: {error_msg}")
#         if einfo:
#             try:
#                 logger.error(f"Exception info: {einfo}")
#             except:
#                 pass
#         if traceback:
#             try:
#                 tb_str = str(traceback)[:2000] if traceback else ""
#                 logger.error(f"Traceback:\n{tb_str}")
#             except:
#                 logger.error("Could not format traceback")
#         logger.error("=" * 100)
        
#         # Try to store a safe error result directly in Redis
#         try:
#             from celery.result import AsyncResult
#             result = AsyncResult(task_id, app=celery_app)
            
#             # Create a safe, JSON-serializable error dict
#             safe_error = {
#                 'success': False,
#                 'error': error_msg[:500],
#                 'error_type': error_type,
#                 'message': f'Task failed: {error_msg[:500]}',
#                 'progress': 100
#             }
            
#             # Verify it's JSON-serializable
#             try:
#                 json.dumps(safe_error)
#             except Exception as json_error:
#                 logger.error(f"Error dict not JSON-serializable: {json_error}")
#                 safe_error = {
#                     'success': False,
#                     'error': 'Task failed (error details could not be serialized)',
#                     'error_type': 'Exception',
#                     'message': 'Task failed',
#                     'progress': 100
#                 }
            
#             # Try to store it directly using the backend
#             try:
#                 backend = celery_app.backend
#                 backend.store_result(
#                     task_id,
#                     safe_error,
#                     'FAILURE'
#                 )
#                 logger.info(f"Stored safe error result for task {task_id}")
#             except Exception as store_error:
#                 logger.warning(f"Could not store error result for task {task_id}: {store_error}")
#                 try:
#                     result.forget()
#                 except:
#                     pass
#         except Exception as e:
#             logger.error(f"Error in task_failure_handler for {task_id}: {e}")
#     except Exception as e:
#         logger.error(f"Critical error in task_failure_handler: {e}")


# Ensure tasks are discovered
celery_app.autodiscover_tasks(['app'])