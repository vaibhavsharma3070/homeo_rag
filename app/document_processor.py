import os
import re
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
                    'chunk_overlap': settings.chunk_overlap
                }
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
