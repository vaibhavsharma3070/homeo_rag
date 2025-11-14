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
import pandas as pd

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
    
    def extract_data_from_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Extract data from CSV file and return as list of row dictionaries."""
        try:
            # Read CSV file with keep_default_na=False to preserve empty strings
            df = pd.read_csv(csv_path, encoding='utf-8', keep_default_na=False, na_values=[''])
            
            # Convert each row to a dictionary
            rows = []
            for idx, row in df.iterrows():
                row_dict = {}
                for col, value in row.items():
                    # Preserve all values including zeros and empty strings
                    # Only skip if truly NaN
                    if pd.isna(value):
                        row_dict[col] = ''
                    else:
                        row_dict[col] = value
                rows.append(row_dict)
            
            logger.info(f"Extracted {len(rows)} rows with {len(df.columns)} columns from {csv_path.name}")
            return rows
            
        except UnicodeDecodeError:
            # Try with different encodings
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, keep_default_na=False, na_values=[''])
                    rows = []
                    for idx, row in df.iterrows():
                        row_dict = {}
                        for col, value in row.items():
                            if pd.isna(value):
                                row_dict[col] = ''
                            else:
                                row_dict[col] = value
                        rows.append(row_dict)
                    logger.info(f"Extracted {len(rows)} rows from {csv_path.name} using {encoding} encoding")
                    return rows
                except:
                    continue
            raise Exception(f"Could not decode CSV file {csv_path.name} with any encoding")
        except Exception as e:
            logger.error(f"Error extracting data from {csv_path}: {e}")
            raise
    
    def extract_data_from_xlsx(self, xlsx_path: Path, sheet_name: str = None) -> List[Dict[str, Any]]:
        """Extract data from XLSX file and return as list of row dictionaries."""
        try:
            # Read XLSX file with na_values=[] to preserve empty strings
            if sheet_name:
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine='openpyxl', keep_default_na=False, na_values=[''])
            else:
                # Read first sheet by default
                df = pd.read_excel(xlsx_path, engine='openpyxl', keep_default_na=False, na_values=[''])
            
            # Convert each row to a dictionary
            rows = []
            for idx, row in df.iterrows():
                row_dict = {}
                for col, value in row.items():
                    # Preserve all values including zeros and empty strings
                    if pd.isna(value):
                        row_dict[col] = ''
                    else:
                        row_dict[col] = value
                rows.append(row_dict)
            
            logger.info(f"Extracted {len(rows)} rows with {len(df.columns)} columns from {xlsx_path.name}")
            return rows
            
        except Exception as e:
            logger.error(f"Error extracting data from {xlsx_path}: {e}")
            raise
    
    def format_row_as_chunk(self, row, row_number=None) -> str:
        parts = []

        if row_number:
            parts.append(f"Record Number: {row_number}")

        for key, value in row.items():
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except:
                pass

            v = str(value).strip()
            if not v or v.lower() in ("none", "nan"):
                continue

            parts.append(f"{key}: {v}")

        return "\n".join(parts)

    
    def verify_csv_processing(self, csv_path: Path) -> Dict[str, Any]:
        """Verify that all rows from CSV are processed correctly."""
        # Read original CSV
        df = pd.read_csv(csv_path, keep_default_na=False)
        original_row_count = len(df)
        
        # Process document
        doc_info = self._process_csv(csv_path)
        processed_row_count = doc_info['total_chunks']
        
        # Compare
        verification = {
            'original_rows': original_row_count,
            'processed_rows': processed_row_count,
            'match': original_row_count == processed_row_count,
            'difference': original_row_count - processed_row_count,
            'accuracy_percentage': (processed_row_count / original_row_count * 100) if original_row_count > 0 else 0
        }
        
        logger.info(f"Verification: {verification}")
        return verification

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
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a document (PDF, CSV, or XLSX) and return structured data."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                return self._process_pdf(file_path)
            elif file_ext == '.csv':
                return self._process_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                return self._process_xlsx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF document and return structured data."""
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
            'source_type': 'pdf'
        }
        
        logger.info(f"Successfully processed {pdf_path.name} into {len(chunks)} chunks")
        return document_info
    
    def _process_csv(self, csv_path: Path) -> Dict[str, Any]:
        """Process a CSV file and return structured data with each row as a chunk."""
        # Extract rows
        rows = self.extract_data_from_csv(csv_path)
        
        if not rows:
            logger.warning(f"No rows extracted from {csv_path.name}")
            return None
        
        # Format each row as a chunk with row tracking
        chunks = []
        skipped_rows = []
        
        for idx, row in enumerate(rows, start=1):  # Start from 1 for human-readable row numbers
            chunk_text = self.format_row_as_chunk(row, row_number=idx)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            else:
                skipped_rows.append(idx)
                logger.warning(f"Skipped empty row {idx} in {csv_path.name}")
        
        # Log statistics
        logger.info(f"CSV Processing Stats - Total rows: {len(rows)}, "
                    f"Valid chunks: {len(chunks)}, Skipped: {len(skipped_rows)}")
        
        if skipped_rows:
            logger.warning(f"Skipped row numbers: {skipped_rows[:10]}..." if len(skipped_rows) > 10 else f"Skipped row numbers: {skipped_rows}")
        
        # Create document metadata
        document_info = {
            'filename': csv_path.name,
            'file_path': str(csv_path),
            'total_chunks': len(chunks),
            'chunks': chunks,
            'metadata': {
                'file_size': csv_path.stat().st_size,
                'processing_timestamp': str(csv_path.stat().st_mtime),
                'source_type': 'csv',
                'total_rows': len(rows),
                'valid_rows': len(chunks),
                'skipped_rows': len(skipped_rows),
                'columns': list(rows[0].keys()) if rows else [],
                'column_count': len(rows[0].keys()) if rows else 0
            },
            'source_type': 'csv'
        }
        
        logger.info(f"Successfully processed {csv_path.name}: {len(chunks)} chunks from {len(rows)} rows")
        return document_info
    
    def _process_xlsx(self, xlsx_path: Path) -> Dict[str, Any]:
        rows = self.extract_data_from_xlsx(xlsx_path)
        
        if not rows:
            logger.warning(f"No rows extracted from {xlsx_path.name}")
            return None

        chunks = []
        skipped_rows = []

        BATCH_SIZE = 5
        batch = []

        for idx, row in enumerate(rows, start=1):
            chunk_text = self.format_row_as_chunk(row, row_number=idx)

            if chunk_text.strip():
                batch.append(chunk_text)
            else:
                skipped_rows.append(idx)
                logger.warning(f"Skipped empty row {idx}")

            # 🔥 When batch reaches 5 rows → make a chunk
            if len(batch) == BATCH_SIZE:
                chunk_block = "\n\n--- RECORD BREAK ---\n\n".join(batch)
                chunks.append(chunk_block)
                batch = []  # reset

        # Handle last leftover rows (<5)
        if batch:
            chunk_block = "\n\n--- RECORD BREAK ---\n\n".join(batch)
            chunks.append(chunk_block)

        document_info = {
            'filename': xlsx_path.name,
            'file_path': str(xlsx_path),
            'chunks': chunks,
            'total_chunks': len(chunks),
            'metadata': {
                'total_rows': len(rows),
                'valid_rows': len(rows) - len(skipped_rows),
                'skipped_rows': skipped_rows,
                'chunk_size': BATCH_SIZE,
            }
        }

        return document_info

    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get list of all processed documents (PDF, CSV, XLSX)."""
        documents = []
        
        # Process PDF files
        for pdf_file in self.upload_dir.glob("*.pdf"):
            try:
                doc_info = self.process_document(pdf_file)
                documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        # Process CSV files
        for csv_file in self.upload_dir.glob("*.csv"):
            try:
                doc_info = self.process_document(csv_file)
                documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue
        
        # Process XLSX files
        for xlsx_file in self.upload_dir.glob("*.xlsx"):
            try:
                doc_info = self.process_document(xlsx_file)
                documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error processing {xlsx_file}: {e}")
                continue
        
        # Process XLS files
        for xls_file in self.upload_dir.glob("*.xls"):
            try:
                doc_info = self.process_document(xls_file)
                documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error processing {xls_file}: {e}")
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