import datetime
import os
import re
import time
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger
from app.config import settings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles PDF document processing, text extraction, and preprocessing."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.processed_dir = Path(settings.processed_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file using PyPDF2 (Windows compatible)."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if chunk_overlap is None:
            chunk_overlap = settings.chunk_overlap
        
        # Use LangChain's RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        logger.info(f"Created {len(chunks)} chunks from text using LangChain")
        return chunks
    
    def process_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF document and return structured data."""
        try:
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Preprocess text
            clean_text = self.preprocess_text(raw_text)
            
            # Create chunks
            chunks = self.chunk_text(clean_text)
            
            # Create document metadata
            document_info = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'total_chunks': len(chunks),
                'chunks': chunks,
                'metadata': {
                    'file_size': pdf_path.stat().st_size,
                    'processing_timestamp': str(pdf_path.stat().st_mtime),
                    'chunk_size': settings.chunk_size,
                    'chunk_overlap': settings.chunk_overlap,
                    'source_type': 'pdf'
                },
                'source_type': 'pdf'  # Add source type at document level
            }
            
            logger.info(f"Successfully processed {pdf_path.name} into {len(chunks)} chunks")
            return document_info
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {e}")
            raise
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents."""
        documents = []
        
        for pdf_file in self.upload_dir.glob("*.pdf"):
            try:
                doc_info = self.process_document(pdf_file)
                documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        return documents

    def process_web_content(self, url: str, content: str, headings: List[str] = None) -> Dict[str, Any]:
        """
        Process web content into chunks for vector storage.
        
        Args:
            url: The source URL
            content: The scraped text content
            headings: List of headings from the page
            
        Returns:
            Dict containing processed document information
        """
        try:
            from urllib.parse import urlparse
            import re
            
            # Extract domain/title from URL for filename
            parsed_url = urlparse(url)
            domain = parsed_url.netloc or "unknown"
            filename = f"{domain}_{int(time.time())}.web"
            
            # Clean and prepare content
            cleaned_content = self._clean_web_content(content)
            
            if not cleaned_content.strip():
                logger.warning(f"No meaningful content found for URL: {url}")
                return None
                
            # Chunk the content using existing chunk_text method
            chunks = self.chunk_text(cleaned_content)  # Fixed: removed underscore
            
            if not chunks:
                logger.warning(f"No chunks created for URL: {url}")
                return None
            
            # Prepare metadata
            metadata = {
                "source_url": url,
                "domain": domain,
                "scraped_at": datetime.datetime.now().isoformat(),
                "content_type": "web_page",
                "headings": headings or [],
                "source_type": "url",
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
            
            doc_info = {
                'filename': filename,
                'file_path': url,  # Use URL as file path for web content
                'chunks': chunks,
                'total_chunks': len(chunks),
                'metadata': metadata,
                'source_type': 'url'  # Add source type at document level
            }
            
            logger.info(f"Web content processed: {len(chunks)} chunks from {url}")
            return doc_info
            
        except Exception as e:
            logger.error(f"Error processing web content from {url}: {e}")
            return None

    def _clean_web_content(self, content: str) -> str:
        """Clean web content for better processing."""
        import re
        
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Copyright \d{4}.*?(?=\.|$)', '', content, flags=re.IGNORECASE)
        
        # Remove navigation elements
        content = re.sub(r'\b(Home|About|Contact|Menu|Navigation|Skip to|Jump to)\b', '', content, flags=re.IGNORECASE)
        
        # Remove Wikipedia-specific artifacts
        content = re.sub(r'\[edit\]|\[citation needed\]|\[\d+\]', '', content)
        content = re.sub(r'From Wikipedia, the free encyclopedia', '', content, flags=re.IGNORECASE)
        
        # Remove common footer/header text
        content = re.sub(r'All rights reserved.*?(?=\.|$)', '', content, flags=re.IGNORECASE)
        
        # Remove URLs
        content = re.sub(r'https?://\S+', '', content)
        
        # Clean up multiple periods and spaces
        content = re.sub(r'\.{2,}', '.', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()