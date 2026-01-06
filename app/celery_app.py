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

# REPLACE the celery_app.conf.update section with this:
celery_app.conf.update(
    task_track_started=True,
    result_expires=300,  # Changed from 3600 to 5 minutes
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    
    task_time_limit=3600,
    task_soft_time_limit=3300,
    
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    
    # ADD THESE NEW SETTINGS
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_disable_rate_limits=True,
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=True,
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

# Add this import at the top
from celery.signals import worker_shutdown

# Add this NEW signal handler (add it after worker_ready_handler)
@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Cleanup on worker shutdown"""
    logger.info("=" * 100)
    logger.info("CELERY WORKER SHUTTING DOWN - CLEANING UP RESOURCES")
    logger.info("=" * 100)
    
    try:
        # Force garbage collection
        import gc
        gc.collect()
        
        # Close any remaining DB connections
        from app.vector_store import PGVectorStore
        if hasattr(PGVectorStore, '_pool'):
            PGVectorStore._pool.closeall()
    except Exception as e:
        logger.error(f"Error during worker shutdown cleanup: {e}")


# UNCOMMENT and UPDATE the task_failure_handler
@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failures properly."""
    try:
        error_msg = str(exception) if exception else "Unknown error"
        error_type = type(exception).__name__ if exception else "Exception"
        
        logger.error("=" * 100)
        logger.error(f"TASK FAILURE: {task_id}")
        logger.error(f"Error: {error_type}: {error_msg}")
        logger.error("=" * 100)
        
        # Clean up the task result in Redis
        from celery.result import AsyncResult
        result = AsyncResult(task_id, app=celery_app)
        try:
            result.forget()  # Remove from backend
        except:
            pass
            
    except Exception as e:
        logger.error(f"Error in task_failure_handler: {e}")


# Ensure tasks are discovered
celery_app.autodiscover_tasks(['app'])