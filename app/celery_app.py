from celery import Celery
from app.config import settings


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
)

# Ensure tasks are discovered inside app.tasks
celery_app.autodiscover_tasks(['app'])


