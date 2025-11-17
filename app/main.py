import uvicorn
from loguru import logger
from app.api import app
from app.config import settings

if __name__ == "__main__":
    logger.info("Starting Homeopathy Knowledgebase RAG API")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"API Host: {settings.api_host}")
    logger.info(f"API Port: {settings.api_port}")
    
    uvicorn.run(
        "app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=300,  # Keep connections alive for 5 minutes
        timeout_graceful_shutdown=30,  # Graceful shutdown timeout
        limit_concurrency=1000,  # Max concurrent connections
        limit_max_requests=10000,  # Max requests before restart
        backlog=2048  # Connection backlog
    )
