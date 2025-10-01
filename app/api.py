import os
import random
import shutil
import asyncio
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

from app.models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    DocumentListResponse, IngestionResponse, StatsResponse, 
    LLMTestResponse, ErrorResponse, ChatSessionResponse, ChatHistoryResponse, ChatMessage
)
from app.rag_pipeline import RAGPipeline
from app.document_processor import DocumentProcessor
from app.config import settings
from app.celery_app import celery_app
from celery.result import AsyncResult

# Global storage for tracking ingestion progress
ingestion_progress_store = {}

class EnhancedIngestionResponse(IngestionResponse):
    """Enhanced response model with parallel processing stats."""
    processing_time: float = 0.0
    chunks_per_second: float = 0.0
    successful_batches: int = 0
    failed_batches: int = 0
    total_batches: int = 0
    max_workers: int = 0
    batch_size: int = 0

# Initialize FastAPI app
app = FastAPI(
    title="Homeopathy Knowledgebase RAG API",
    description="API for homeopathy knowledge retrieval and question answering with parallel processing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize components
rag_pipeline = RAGPipeline()
document_processor = DocumentProcessor()

@app.get("/")
async def serve_ui():
    """Serve the main UI."""
    ui_file = static_dir / "index.html"
    if ui_file.exists():
        return FileResponse(str(ui_file))
    else:
        return {
            "message": "Enhanced Homeopathy Knowledgebase RAG API with Parallel Processing",
            "version": "2.0.0",
            "status": "running",
            "features": ["parallel_ingestion", "batch_processing", "progress_tracking"],
            "endpoints": {
                "ingest": "/api/ingest - Parallel document ingestion",
                "ingest_async": "/api/ingest/async - Async ingestion with progress tracking",
                "ingest_progress": "/api/ingest/progress/{job_id} - Check ingestion progress",
                "query": "/api/query",
                "search": "/api/search",
                "documents": "/api/documents",
                "stats": "/api/stats",
                "llm-test": "/api/llm-test",
                "scrape-test": "/api/scrape-test - Test web scraping functionality",
                "weblink": "/api/weblink - Scrape and ingest web content"
            }
        }

@app.post("/api/ingest", response_model=EnhancedIngestionResponse, tags=["Documents"])
async def ingest_documents_parallel(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    max_workers: int = Form(default=4, description="Number of parallel workers"),
    batch_size: int = Form(default=100, description="Chunks per batch")
):
    """Ingest PDF documents using parallel processing."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF"
                )
        
        processed_docs = []
        total_chunks = 0
        
        # Process all files first
        logger.info(f"Starting document processing for {len(files)} files...")
        for file in files:
            try:
                # Save uploaded file
                file_path = Path(settings.upload_dir) / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process document
                doc_info = document_processor.process_document(file_path)
                processed_docs.append(doc_info)
                total_chunks += doc_info['total_chunks']
                
                logger.info(f"Processed {file.filename} into {doc_info['total_chunks']} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                continue
        
        if not processed_docs:
            raise HTTPException(status_code=500, detail="No documents were successfully processed")
        
        # Configure parallel processing
        rag_pipeline.vector_store.max_workers = max_workers
        rag_pipeline.vector_store.batch_size = batch_size
        
        logger.info(f"Starting parallel ingestion with {max_workers} workers, batch size {batch_size}")
        
        # Add documents using parallel processing
        result_stats = rag_pipeline.vector_store.add_documents_parallel(processed_docs)
        
        if result_stats['success']:
            return EnhancedIngestionResponse(
                success=True,
                message=f"Successfully ingested {len(processed_docs)} documents using parallel processing",
                documents_processed=result_stats['total_documents'],
                chunks_created=result_stats['processed_chunks'],
                processing_time=result_stats['processing_time'],
                chunks_per_second=result_stats['chunks_per_second'],
                successful_batches=result_stats['successful_batches'],
                failed_batches=result_stats['failed_batches'],
                total_batches=result_stats['total_batches'],
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Partial failure: {result_stats['failed_chunks']} chunks failed to process"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in parallel document ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/ingest/async", tags=["Documents"])
async def ingest_documents_async(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    max_workers: int = Form(default=4),
    batch_size: int = Form(default=100)
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        saved_paths = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            dest = Path(settings.upload_dir) / file.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(dest))

        # Enqueue celery job
        task = celery_app.send_task(
            "app.tasks.ingest_documents_task",
            args=[saved_paths, max_workers, batch_size]
        )
        return {"job_id": task.id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling async ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingest/progress/{job_id}", tags=["Documents"])
async def get_ingest_progress(job_id: str):
    try:
        res = AsyncResult(job_id, app=celery_app)
        response = {
            "job_id": job_id,
            "state": res.state,
            "progress": 0,
            "detail": None,
        }
        info = res.info or {}
        if isinstance(info, dict):
            response.update(info)
        if res.successful():
            response["state"] = "SUCCESS"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def process_query(request: QueryRequest):
    """Process a query through the RAG pipeline."""
    try:
        # Process, passing session_id to include conversational history in context
        result = rag_pipeline.process_query(request.query, request.top_k, session_id=request.session_id)
        # print('here is the response =====', result)
        # Persist chat messages if database available
        try:
            if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'save_chat_message'):
                if request.session_id:
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'user', request.query)
                    answer_text = str(result.get('answer', ''))
                    rag_pipeline.vector_store.save_chat_message(request.session_id, 'ai', answer_text)
        except Exception as persist_err:
            logger.warning(f"Failed to save chat history: {persist_err}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/chat/new", response_model=ChatSessionResponse, tags=["RAG"])
async def create_chat_session():
    """Create and return a new chat session id."""
    try:
        # Simple session id generator; replace with robust server-side tracking if needed
        import uuid
        session_id = uuid.uuid4().hex
        return ChatSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating chat session: {str(e)}")

@app.get("/api/chat/{session_id}", response_model=ChatHistoryResponse, tags=["RAG"])
async def get_chat_history(session_id: str):
    """Return stored chat history for a given session."""
    try:
        if hasattr(rag_pipeline, 'vector_store') and hasattr(rag_pipeline.vector_store, 'get_chat_history'):
            rows = rag_pipeline.vector_store.get_chat_history(session_id)
            messages = [ChatMessage(id=r.get('id'), session_id=session_id, role=r.get('role'), message=r.get('message')) for r in rows]
            return ChatHistoryResponse(session_id=session_id, messages=messages)
        else:
            return ChatHistoryResponse(session_id=session_id, messages=[])
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        doc_list = rag_pipeline.get_all_documents()
        if doc_list is None:
            doc_list = []
        return DocumentListResponse(documents=doc_list, total_documents=len(doc_list))
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

def scrape_url(url: str, depth: int = 1, visited=None, use_proxy: bool = False, proxy_list: List[str] = None):
    if visited is None:
        visited = set()
    if url in visited:
        return []
    visited.add(url)
    results = []

    # Create robust session with retry mechanism
    session = requests.Session()

    # Rotate between different realistic user agents
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ]

    # Enhanced headers with more realistic browser behavior
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
        'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
    })

    session.max_redirects = 5
    session.cookies.clear()

    # Set a realistic referer for the domain
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc:
            session.headers['Referer'] = f"https://{parsed_url.netloc}/"
    except Exception:
        pass

    # Configure proxy if enabled
    if use_proxy and proxy_list:
        try:
            proxy = random.choice(proxy_list)
            session.proxies = {'http': proxy, 'https': proxy}
            logger.info(f"Using proxy: {proxy}")
        except Exception as e:
            logger.warning(f"Failed to configure proxy: {e}")

    # Retry mechanism with exponential backoff
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            delay = random.uniform(base_delay, base_delay + 2) + (attempt * 2)
            time.sleep(delay)

            if attempt > 0:
                session.headers['User-Agent'] = random.choice(user_agents)
                session.headers['Accept-Language'] = random.choice([
                    'en-US,en;q=0.9',
                    'en-US,en;q=0.9,es;q=0.8',
                    'en-GB,en;q=0.9,en-US;q=0.8'
                ])

            res = session.get(url, timeout=20, allow_redirects=True)

            if res.status_code == 403:
                logger.warning(f"403 Forbidden for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                results.append({"url": url, "error": f"403 Forbidden after {max_retries} attempts"})
                return results

            elif res.status_code == 429:
                logger.warning(f"Rate limited for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(10 + (attempt * 5))
                    continue
                results.append({"url": url, "error": f"Rate limited after {max_retries} attempts"})
                return results

            elif res.status_code == 200:
                res.raise_for_status()
                break
            else:
                res.raise_for_status()

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            results.append({"url": url, "error": f"Request failed after {max_retries} attempts: {str(e)}"})
            return results

    # If we get here, request was successful
    try:
        soup = BeautifulSoup(res.text, "html.parser")

        for unwanted in soup(["script", "style", "noscript"]):
            unwanted.decompose()

        for unwanted in soup(["nav", "header", "footer", "aside", "advertisement", "ads",
                              "social-share", "cookie-banner", "iframe"]):
            text_content = unwanted.get_text(strip=True)
            if text_content and len(text_content) > 20:
                temp_div = soup.new_tag("div", **{"class": "extracted-ui-text", "data-source": unwanted.name})
                temp_div.string = f"UI_ELEMENT_{unwanted.name.upper()}: {text_content}"
                soup.append(temp_div)
            unwanted.decompose()

        # Extract different elements
        headings, heading_hierarchy, paragraphs, div_content = [], [], [], []
        semantic_content, list_content, table_content = [], [], []
        form_content, span_content, code_content = [], [], []
        other_content, data_content, meta_content, selector_content = [], [], [], []
        remaining_text = []

        # Headings
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            heading_text = h.get_text(strip=True)
            if heading_text and len(heading_text) > 2:
                headings.append(heading_text)
                level = int(h.name[1])
                heading_hierarchy.append(f"{'  ' * (level-1)}• {heading_text}")

        # Paragraphs
        for p in soup.find_all("p"):
            para_text = p.get_text(strip=True)
            if para_text and len(para_text) > 10:
                paragraphs.append(para_text)

        # Divs
        for div in soup.find_all("div"):
            div_class = " ".join(div.get("class", [])).lower()
            div_id = div.get("id", "").lower()
            if any(skip_word in div_class or skip_word in div_id for skip_word in
                   ["nav", "menu", "sidebar", "ad", "advertisement", "banner", "footer",
                    "header", "social", "share", "comment", "cookie", "popup", "modal"]):
                continue
            div_text = div.get_text(strip=True)
            if div_text and len(div_text) > 20:
                div_content.append(div_text)

        # Semantic
        for tag in soup.find_all(["main", "article", "section", "aside", "details", "summary",
                                  "blockquote", "cite", "q", "mark", "ins", "del", "s", "u"]):
            content_text = tag.get_text(strip=True)
            if content_text and len(content_text) > 15:
                semantic_content.append(content_text)

        # Lists
        for list_tag in soup.find_all(["ul", "ol", "dl"]):
            list_text = list_tag.get_text(strip=True)
            if list_text and len(list_text) > 20:
                list_content.append(list_text)

        # Tables
        for table in soup.find_all("table"):
            table_text = table.get_text(strip=True)
            if table_text and len(table_text) > 30:
                table_content.append(table_text)

        # Forms
        for form in soup.find_all("form"):
            form_text = form.get_text(strip=True)
            if form_text and len(form_text) > 20:
                form_content.append(form_text)

        # Spans
        for span in soup.find_all("span"):
            span_text = span.get_text(strip=True)
            if span_text and len(span_text) > 10:
                span_content.append(span_text)

        # Code blocks
        for code_tag in soup.find_all(["pre", "code", "kbd", "samp", "var"]):
            code_text = code_tag.get_text(strip=True)
            if code_text and len(code_text) > 10:
                code_content.append(code_text)

        # Other text
        for tag in soup.find_all(["strong", "b", "em", "i", "small", "sub", "sup",
                                  "abbr", "acronym", "address", "bdo", "big", "tt"]):
            tag_text = tag.get_text(strip=True)
            if tag_text and len(tag_text) > 5:
                other_content.append(tag_text)

        # Data attributes
        for element in soup.find_all(attrs={"data-content": True}):
            data_text = element.get("data-content", "").strip()
            if data_text and len(data_text) > 10:
                data_content.append(data_text)

        # Title + meta
        title = soup.find('title')
        if title and title.get_text(strip=True):
            meta_content.append(f"Page Title: {title.get_text(strip=True)}")

        for meta in soup.find_all('meta', attrs={'name': ['description', 'keywords', 'author', 'subject']}):
            content = meta.get('content', '').strip()
            if content:
                meta_content.append(f"{meta.get('name', 'meta')}: {content}")

        # Content selectors
        content_selectors = [
            '.content', '.main-content', '.article-content', '.post-content',
            '.entry-content', '.page-content', '.text-content', '.body-content',
            '.content-body', '.main-body', '.article-body', '.post-body',
            '.content-area', '.main-area', '.primary-content', '.secondary-content',
            '[role="main"]', '[role="article"]', '[role="contentinfo"]',
            '.container', '.wrapper', '.inner', '.content-wrapper'
        ]
        for selector in content_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:
                        selector_content.append(f"SELECTOR_{selector}: {text}")
            except Exception:
                continue

        # Remaining text
        for element in soup.find_all(text=True):
            if element.parent and element.parent.name not in ['script', 'style', 'noscript']:
                text = element.strip()
                if text and len(text) > 5:
                    remaining_text.append(text)

        # Organize by priority
        high_priority_content = [
            ("Headings", headings),
            ("Paragraphs", paragraphs),
            ("Semantic Content", semantic_content),
            ("Table Content", table_content),
            ("Content Selectors", selector_content)
        ]
        medium_priority_content = [
            ("List Content", list_content),
            ("Code Content", code_content),
            ("Meta Information", meta_content),
            ("Data Attributes", data_content)
        ]
        low_priority_content = [
            ("Div Content", div_content),
            ("Form Content", form_content),
            ("Span Content", span_content),
            ("Other Text Elements", other_content),
            ("Remaining Text", remaining_text)
        ]

        all_text_parts = []
        for priority_section, content_groups in [
            ("HIGH PRIORITY CONTENT", high_priority_content),
            ("MEDIUM PRIORITY CONTENT", medium_priority_content),
            ("LOW PRIORITY CONTENT", low_priority_content)
        ]:
            all_text_parts.append(f"\n{'='*20} {priority_section} {'='*20}")
            for content_name, content_list in content_groups:
                if content_list:
                    all_text_parts.append(f"\n--- {content_name.upper()} ---")
                    all_text_parts.extend(content_list)

        all_text = "\n".join(all_text_parts)

        # Cleanup
        all_text = re.sub(r'\n\s*\n', '\n\n', all_text)
        all_text = re.sub(r'[ \t]+', ' ', all_text)
        for pattern in [
            r'Cookie Policy|Privacy Policy|Terms of Service|Subscribe|Newsletter|Sign up|Login|Register',
            r'From Wikipedia, the free encyclopedia|Jump to navigation|Jump to search',
            r'Skip to main content|Skip to navigation|Skip to search',
            r'Home|About|Contact|Support|Help|FAQ',
            r'Follow us|Share|Like|Tweet|Share on|Follow on',
            r'Advertisement|Ad|Sponsored|Promoted',
            r'© \d{4}|All rights reserved|Copyright',
            r'Last updated|Last modified|Published on|Posted on',
            r'Read more|Continue reading|Show more|View more',
            r'Loading|Please wait|Error|Warning|Notice',
            r'JavaScript is required|Enable JavaScript|Update your browser',
            r'This site uses cookies|Accept cookies|Cookie settings'
        ]:
            all_text = re.sub(pattern, '', all_text, flags=re.IGNORECASE)

        all_text = re.sub(r'\n{3,}', '\n\n', all_text).strip()

        if all_text and len(all_text) > 50:
            results.append({
                "url": url,
                "headings": headings,
                "text": all_text
            })
            logger.info(f"Successfully scraped {url}: {len(all_text)} chars, {len(headings)} headings")
        else:
            logger.warning(f"Insufficient content scraped from {url}")

        # Depth handling
        if depth > 1:
            base_domain = urlparse(url).netloc
            links = soup.find_all("a", href=True)
            followed_links = 0
            for a in links:
                if followed_links >= 5:
                    break
                child_url = urljoin(url, a["href"])
                if (child_url.startswith("http") and
                        urlparse(child_url).netloc == base_domain and
                        not any(child_url.lower().endswith(ext) for ext in
                                ['.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.xml', '.json'])):
                    try:
                        child_results = scrape_url(child_url, depth - 1, visited)
                        results.extend(child_results)
                        followed_links += 1
                    except Exception as e:
                        logger.warning(f"Failed to scrape child URL {child_url}: {e}")
                        continue

    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        results.append({"url": url, "error": str(e)})
    finally:
        session.close()

    return results


@app.post("/api/weblink", tags=["Weblink"])
async def weblink_endpoint(request: Dict[str, Any]):
    """
    Schedule weblink scraping and ingestion as a background Celery task.
    Returns a job_id to poll with any Celery-aware progress endpoint.
    
    Optional parameters:
    - use_proxy: bool - Enable proxy usage for scraping
    - proxy_list: List[str] - List of proxy URLs to use
    """
    try:
        url = request.get("query")
        depth = int(request.get("depth", 1))
        max_workers = int(request.get("max_workers", 4))
        batch_size = int(request.get("batch_size", 100))
        use_proxy = request.get("use_proxy", False)
        proxy_list = request.get("proxy_list", [])

        if not url:
            raise HTTPException(status_code=400, detail="No URL provided")

        task = celery_app.send_task(
            "app.tasks.weblink_ingestion_task",
            args=[url, depth, max_workers, batch_size, use_proxy, proxy_list]
        )
        return {"job_id": task.id, "status": "queued", "use_proxy": use_proxy}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scheduling weblink async ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scrape-test", tags=["System"])
async def test_scraping(url: str = "https://httpbin.org/user-agent", use_proxy: bool = False):
    """Test web scraping functionality with a simple URL."""
    try:
        result = scrape_url(url, depth=1, use_proxy=use_proxy)
        
        if result and len(result) > 0:
            if 'error' in result[0]:
                return {
                    "success": False,
                    "error": result[0]['error'],
                    "url": url,
                    "use_proxy": use_proxy
                }
            else:
                return {
                    "success": True,
                    "url": url,
                    "content_length": len(result[0].get('text', '')),
                    "headings_count": len(result[0].get('headings', [])),
                    "use_proxy": use_proxy,
                    "preview": result[0].get('text', '')[:200] + "..." if len(result[0].get('text', '')) > 200 else result[0].get('text', '')
                }
        else:
            return {
                "success": False,
                "error": "No content scraped",
                "url": url,
                "use_proxy": use_proxy
            }
            
    except Exception as e:
        logger.error(f"Error testing scraping: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "use_proxy": use_proxy
        }

@app.get("/api/llm-test", response_model=LLMTestResponse, tags=["System"])
async def test_llm():
    """Test LLM connection and functionality."""
    try:
        result = rag_pipeline.test_llm_connection()
        print(result)
        return LLMTestResponse(**result)
        
    except Exception as e:
        logger.error(f"Error testing LLM: {e}")
        return LLMTestResponse(
            status="error",
            provider="unknown",
            model="unknown",
            error=str(e)
        )

@app.delete("/api/documents", tags=["Documents"])
async def clear_knowledge_base():
    """Clear the entire knowledge base."""
    try:
        success = rag_pipeline.clear_knowledge_base()
        
        if success:
            return {"message": "Knowledge base cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")

@app.get("/api/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get enhanced knowledge base statistics including parallel processing info."""
    try:
        stats = rag_pipeline.get_knowledge_base_stats()
        
        if 'error' in stats:
            raise HTTPException(status_code=500, detail=stats['error'])
        
        # Add parallel processing stats if available
        if hasattr(rag_pipeline.vector_store, 'max_workers'):
            stats.update({
                'parallel_processing': {
                    'max_workers': rag_pipeline.vector_store.max_workers,
                    'batch_size': rag_pipeline.vector_store.batch_size,
                    'enabled': True
                }
            })
        
        return StatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/performance/benchmark", tags=["System"])
async def benchmark_ingestion():
    """Benchmark different ingestion configurations."""
    try:
        # This would run a benchmark with different worker/batch combinations
        # For demo purposes, returning mock data
        benchmark_results = {
            "configurations": [
                {
                    "workers": 1,
                    "batch_size": 50,
                    "chunks_per_second": 45.2,
                    "total_time": 120.5
                },
                {
                    "workers": 2,
                    "batch_size": 100,
                    "chunks_per_second": 78.1,
                    "total_time": 69.8
                },
                {
                    "workers": 4,
                    "batch_size": 100,
                    "chunks_per_second": 142.3,
                    "total_time": 38.3
                },
                {
                    "workers": 8,
                    "batch_size": 200,
                    "chunks_per_second": 201.7,
                    "total_time": 27.1
                }
            ],
            "recommended": {
                "workers": 4,
                "batch_size": 100,
                "reason": "Best balance of speed and resource usage"
            }
        }
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Error running benchmark: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            details={"exception": str(exc)}
        ).dict()
    )