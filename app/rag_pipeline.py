from typing import List, Dict, Any, Optional
from loguru import logger
from app.vector_store import PGVectorStore
from app.llm_connector import LLMFactory, OllamaConnector
from app.config import settings
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval and generation."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: OllamaConnector = LLMFactory.create_connector()
        logger.info("RAG Pipeline initialized")

    def process_query(self, query: str, top_k: int = 3, min_score: float = 0.2) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {query[:100]}...")
            search_results = self.vector_store.search(query, top_k=top_k)
            logger.info(f"Retrieved {len(search_results)} search results")

            if not search_results:
                return self._empty_response(query, "I don't have information about this topic.")

            filtered_results = [r for r in search_results if r['score'] >= min_score]
            print('filer :- ',filtered_results)
            logger.info(f"After filtering: {len(filtered_results)} results remain")

            if not filtered_results:
                return self._empty_response(query, "I don't have enough relevant information about this topic.")

            context_chunks = self._prepare_context(filtered_results, query)
            sources = self._create_sources(filtered_results)

            logger.info("Generating response with LLM...")
            answer = self._generate_response(query, context_chunks)
            logger.info(f"LLM response generated: {len(answer)} characters")

            avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
            # confidence = self._calculate_confidence(avg_score)
            confidence = self._calculate_confidence(avg_score, query, context_chunks)

            return {
                'query': query,
                'answer': answer,
                'context_used': context_chunks,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'total_sources': len(sources),
                    'avg_relevance_score': round(avg_score, 4),
                    'llm_provider': self.llm_connector.__class__.__name__
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_response(query, str(e))

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Dynamically extract keywords:
        - Removes stopwords
        - Preserves acronyms (all-uppercase like IT, AI)
        - Keeps words > 2 chars
        """
        words = query.replace("?", "").split()
        keywords = []
        for w in words:
            wl = w.lower()
            if w.isupper():
                keywords.append(w)  # keep acronyms
            elif wl not in STOPWORDS and len(wl) > 2:
                keywords.append(wl)
        return keywords

    def _prepare_context(self, results: List[Dict[str, Any]], query: str, max_chunks: int = 5) -> List[str]:
        """
        Build context from search results.
        - Check all results.
        - Rank by keyword overlap with the query (case-insensitive).
        - Return top `max_chunks` relevant sentences.
        """
        # query_keywords = [word.lower() for word in query.split() if len(word) > 2]
        query_keywords = self._extract_keywords(query)
        print('query_keywords :- ',query_keywords)
        ranked_contexts = []

        for result in results:
            text = result['text'].replace('--- Page', '').replace('---', '').strip()
            text = ' '.join(text.split())
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]

            # Score sentences by keyword overlap (case-insensitive)
            for s in sentences:
                score = sum(k in s.lower() for k in query_keywords)
                if score > 0:
                    ranked_contexts.append((score, s))

        ranked_contexts.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in ranked_contexts[:max_chunks]]

        return [". ".join(top_sentences) + "."] if top_sentences else []


    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for r in results:
            src = {
                'filename': r['filename'],
                'document_id': r['document_id'],
                'relevance_score': round(r['score'], 4)
            }
            if r.get('chunk_id'):
                src['chunk_id'] = r['chunk_id']
            text = ' '.join(r['text'].replace('--- Page', '').replace('---', '').split())
            src['preview'] = text[:80] + "..." if len(text) > 80 else text
            sources.append(src)
        return sources

    def _generate_response(self, query: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "I don't have information about this."

        # Combine all chunks with numbering for clarity
        context_text = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(context_chunks)])
        
        prompt = f"""
        You are a helpful assistant. Read the following information carefully and answer the question.
        Summarize the answer clearly, concisely, and completely.

        Information:
        {context_text}

        Question:
        {query}

        Answer (provide a clear summary):
        """

        response = self.llm_connector.generate_response(prompt)

        # fallback to dynamic summarization if LLM gives too short response
        if len(response.strip()) < 20:
            from itertools import islice
            # join top 5-10 sentences dynamically
            sentences = " ".join(context_chunks).split('. ')
            response = ". ".join(islice(sentences, 0, min(10, len(sentences)))) + "."
        
        return response.strip()



    def _calculate_confidence(self, avg_score: float, query: str = "", context_chunks: List[str] = None) -> str:
        """
        Compute confidence level using avg_score + keyword presence in context.
        - Case-insensitive keyword matching.
        - Boosts confidence if exact query terms appear in context.
        """
        if avg_score > 0.6:
            return "high"
        elif avg_score > 0.4:
            return "medium"

        # Keyword presence check
        if query and context_chunks:
            # query_keywords = [w.lower() for w in query.split() if len(w) > 2]
            query_keywords = self._extract_keywords(query)
            context_text = " ".join(context_chunks).lower()
            if any(kw in context_text for kw in query_keywords):
                return "medium"

        if avg_score > 0.2:
            return "low"
        return "very_low"


    def _empty_response(self, query: str, msg: str) -> Dict[str, Any]:
        return {
            'query': query,
            'answer': msg,
            'context_used': [],
            'sources': [],
            'confidence': 'low',
            'metadata': {
                'total_sources': 0,
                'avg_relevance_score': 0.0,
                'llm_provider': self.llm_connector.__class__.__name__
            }
        }

    def _error_response(self, query: str, err: str) -> Dict[str, Any]:
        return {
            'query': query,
            'answer': f"An error occurred while processing your query: {err}",
            'context_used': [],
            'sources': [],
            'confidence': 'error',
            'metadata': {
                'total_sources': 0,
                'avg_relevance_score': 0.0,
                'llm_provider': self.llm_connector.__class__.__name__,
                'error': err
            }
        }
    
    # Keep existing utility methods unchanged
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add new documents to the knowledge base."""
        try:
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"Successfully added {len(documents)} documents to knowledge base")
            return success
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 10, min_score: float = 0.2) -> List[Dict[str, Any]]:
        """Search documents without generating a response."""
        try:
            results = self.vector_store.search(query, top_k=top_k)
            return [r for r in results if r['score'] >= min_score]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = self.vector_store.get_index_stats()
            stats['llm_available'] = self.llm_connector.is_available()
            stats['llm_provider'] = self.llm_connector.__class__.__name__
            return stats
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}
    
    def get_all_documents(self) -> Dict[str,Any]:
        """Get all documents"""
        return self.vector_store.get_all_documents()


    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        return self.vector_store.get_document_by_id(doc_id)
    
    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        return self.vector_store.get_document_chunks(doc_id)
    
    def clear_knowledge_base(self) -> bool:
        """Clear the entire knowledge base."""
        try:
            self.vector_store.clear_index()
            logger.info("Knowledge base cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
    
    def test_llm_connection(self) -> Dict[str, Any]:
        """Test the LLM connection and return status."""
        try:
            is_available = self.llm_connector.is_available()
            
            if is_available:
                # Test with a simple query
                test_response = self.llm_connector.generate_response("Hello")
                test_successful = len(test_response) > 0 and "error" not in test_response.lower()
                
                return {
                    'status': 'connected',
                    'provider': self.llm_connector.__class__.__name__,
                    'test_successful': test_successful,
                    'test_response': test_response[:100] + "..." if len(test_response) > 100 else test_response
                }
            else:
                return {
                    'status': 'disconnected',
                    'provider': self.llm_connector.__class__.__name__,
                    'error': 'LLM service not available'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'provider': self.llm_connector.__class__.__name__,
                'error': str(e)
            }