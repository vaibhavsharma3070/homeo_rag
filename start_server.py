#!/usr/bin/env python3
"""
Startup script for the Homeopathy Knowledgebase RAG API server.
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.config import settings
from app.utils import create_directory_structure, log_system_info

def setup_environment():
    """Set up the environment and create necessary directories."""
    logger.info("Setting up environment...")
    
    # Create base data directory
    base_path = Path("./data")
    base_path.mkdir(exist_ok=True)
    
    # Create directory structure
    success = create_directory_structure(base_path)
    if not success:
        logger.error("Failed to create directory structure")
        return False
    
    # Log system information
    log_system_info()
    
    return True

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'faiss-cpu',
        'sentence-transformers',
        'torch',
        'transformers',
        'PyPDF2',
        'loguru'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available")
    return True

def check_llm_availability():
    """Check LLM service availability."""
    logger.info("Checking LLM service availability...")
    
    try:
        from app.llm_connector import LLMFactory
        
        connector = LLMFactory.create_connector()
        is_available = connector.is_available()
        
        if is_available:
            logger.info(f"LLM service is available: {connector.__class__.__name__}")
            return True
        else:
            logger.warning(f"LLM service is not available: {connector.__class__.__name__}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking LLM availability: {e}")
        return False

def start_server(host=None, port=None, reload=None):
    """Start the FastAPI server."""
    import uvicorn
    
    # Use command line arguments or fall back to settings
    host = host or settings.api_host
    port = port or settings.api_port
    reload = reload if reload is not None else settings.debug
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload mode: {reload}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        uvicorn.run(
            "app.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level=settings.log_level.lower(),
            timeout_keep_alive=300,  # Keep connections alive for 5 minutes
            timeout_graceful_shutdown=30,  # Graceful shutdown timeout
            limit_concurrency=1000,  # Max concurrent connections
            limit_max_requests=10000,  # Max requests before restart
            backlog=2048  # Connection backlog
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start Homeopathy Knowledgebase RAG API")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("=" * 60)
    logger.info("Homeopathy Knowledgebase RAG API Server")
    logger.info("=" * 60)
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Check LLM availability
    if not check_llm_availability():
        logger.warning("LLM service not available - some features may not work")
    
    if args.check_only:
        logger.info("Dependency check completed successfully")
        return
    
    # Determine reload setting
    reload = None
    if args.reload:
        reload = True
    elif args.no_reload:
        reload = False
    
    # Start server
    start_server(args.host, args.port, reload)

if __name__ == "__main__":
    main()
