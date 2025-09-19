# Homeopathy Knowledgebase RAG Pipeline

from importlib import import_module

try:
    celery_app = import_module('app.celery_app').celery_app  # type: ignore
except Exception:
    celery_app = None