import os
import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""

def validate_pdf_file(file_path: Path) -> bool:
    """Validate if a file is a valid PDF."""
    try:
        # Check file extension
        if not file_path.suffix.lower() == '.pdf':
            return False
        
        # Check if file exists and is readable
        if not file_path.exists() or not file_path.is_file():
            return False
        
        # Check file size (reasonable limits)
        file_size = file_path.stat().st_size
        if file_size < 1024 or file_size > 100 * 1024 * 1024:  # 1KB to 100MB
            return False
        
        # Try to read first few bytes to check PDF header
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating PDF file {file_path}: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    sanitized = filename
    
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized

def create_directory_structure(base_path: Path) -> bool:
    """Create necessary directory structure."""
    try:
        directories = [
            base_path / "uploads",
            base_path / "processed",
            base_path / "faiss_index",
            base_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directory structure created at {base_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        return False

def save_metadata(metadata: Dict[str, Any], file_path: Path) -> bool:
    """Save metadata to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return False

def load_metadata(file_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return {}

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def chunk_text_smart(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Smart text chunking that tries to preserve sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters of the chunk
            search_start = max(start, end - 100)
            search_text = text[search_start:end]
            
            # Find the last sentence ending
            sentence_endings = ['.', '!', '?', '\n']
            last_ending = -1
            for ending in sentence_endings:
                pos = search_text.rfind(ending)
                if pos > last_ending:
                    last_ending = pos
            
            if last_ending != -1:
                end = search_start + last_ending + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position for next chunk, accounting for overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks

def calculate_embedding_dimension(model_name: str) -> int:
    """Get embedding dimension for a given model."""
    # Common embedding dimensions for popular models
    model_dimensions = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    }
    
    return model_dimensions.get(model_name, 768)  # Default to 768

def log_system_info():
    """Log system information for debugging."""
    import platform
    import sys
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Log available memory (if psutil is available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {format_file_size(memory.total)}")
        logger.info(f"Available Memory: {format_file_size(memory.available)}")
    except ImportError:
        logger.info("psutil not available for memory info")
    
    logger.info("========================")
