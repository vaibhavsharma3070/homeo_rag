from typing import List, Dict, Any, Optional
from loguru import logger
from app.vector_store import PGVectorStore
from app.llm_connector import LLMFactory, OllamaConnector
from app.config import settings
from nltk.corpus import stopwords
from nltk import ngrams
import re

STOPWORDS = set(stopwords.words("english"))

class RAGPipeline:
    """RAG pipeline."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: OllamaConnector = LLMFactory.create_connector()

    def process_query(self, query: str, top_k: int = 3, min_score: float = 0.2) -> Dict[str, Any]:
        """
        Flow:
        1. Initial search with original query
        2. Filter by score
        3. Build context and generate answer
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Step 1: Initial search with original query
            initial_results = self.vector_store.search(query, top_k=top_k)
            logger.info(f"Initial search retrieved {len(initial_results)} results")

            # Step 2: Filter results
            filtered_initial = [r for r in initial_results if r['score'] >= min_score]
            print(f"filtered_initial :- {filtered_initial[:2] if len(filtered_initial) > 2 else filtered_initial}")
            logger.info(f"After filtering: {len(filtered_initial)} results remain")

            search_results = filtered_initial

            # Step 3: Handle no results case
            if not search_results:
                logger.info("No results found for the given query")
                return self._empty_response(
                    query, 
                    f"I don't have information about '{query}'. Please try rephrasing your question or provide more context."
                )

            # Step 4: Prepare context and generate response
            context_chunks = self._prepare_context(search_results, query)
            sources = self._create_sources(search_results)

            logger.info("Generating response with LLM...")
            answer = self._generate_response(query, context_chunks)
            logger.info(f"LLM response generated: {len(answer)} characters")

            avg_score = sum(r['score'] for r in search_results) / len(search_results)
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
                    'llm_provider': self.llm_connector.__class__.__name__,
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_response(query, str(e))

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Dynamic keyword extraction:
        - Detects multi-word technical terms (using n-grams)
        - Removes stopwords
        - Preserves acronyms (IT, AI, API, etc.)
        - Falls back to single words if no n-gram matches
        """
        query_clean = query.lower().replace("?", "").strip()
        words = [w for w in query_clean.split() if w not in STOPWORDS and len(w) > 2]
        
        keywords = []

        # 1. Preserve acronyms (from original query, not lowercased)
        for w in query.split():
            if w.isupper() and len(w) > 1:
                keywords.append(w)

        # 2. Try bigrams and trigrams (multi-word technical phrases)
        bigrams = [" ".join(bg) for bg in ngrams(words, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(words, 3)]

        # Keep n-grams that look meaningful (appear in query as-is)
        meaningful_phrases = [p for p in trigrams + bigrams if p in query_clean]

        if meaningful_phrases:
            keywords.extend(meaningful_phrases)
            logger.info(f"🔍 Found dynamic phrases: {meaningful_phrases}")
        else:
            # 3. Fallback to single-word keywords
            keywords.extend(words)

        # Deduplicate and return
        return list(dict.fromkeys(keywords))

    def _prepare_context(self, results: List[Dict[str, Any]], query: str, max_chunks: int = 5) -> List[str]:
        """
        Build context from search results with improved ranking.
        Now handles multi-word phrases properly.
        """
        query_keywords = self._extract_keywords(query)
        print(f'🔍 Query keywords: {query_keywords}')
        
        ranked_contexts = []

        for result in results:
            text = result['text'].replace('--- Page', '').replace('---', '').strip()
            text = ' '.join(text.split())
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]

            # Score sentences by keyword overlap (case-insensitive)
            for sentence in sentences:
                score = 0
                sentence_lower = sentence.lower()
                
                for keyword in query_keywords:
                    keyword_lower = keyword.lower()
                    
                    # For multi-word phrases, check exact phrase match
                    if " " in keyword_lower:
                        if keyword_lower in sentence_lower:
                            score += 3  # Higher score for phrase match
                    else:
                        # Single word matching
                        if keyword_lower in sentence_lower:
                            # Exact word boundary match gets higher score
                            if f" {keyword_lower} " in f" {sentence_lower} ":
                                score += 2
                            else:
                                score += 1
                
                if score > 0:
                    ranked_contexts.append((score, sentence, result['score']))

        # Sort by keyword score first, then by vector similarity score
        ranked_contexts.sort(key=lambda x: (x[0], x[2]), reverse=True)
        top_sentences = [sentence for _, sentence, _ in ranked_contexts[:max_chunks]]
        
        print(f"📝 Selected {len(top_sentences)} sentences for context")
        return [". ".join(top_sentences) + "."] if top_sentences else []

    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create source information from search results."""
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

    def _generate_response(self, original_query: str, context_chunks: List[str]) -> str:
        """Generate response strictly from provided context."""
        if not context_chunks:
            return "I don't have information about this."

        # Combine all chunks with numbering for clarity
        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        
        prompt = f"""
You are a helpful assistant. Answer ONLY based on the provided information below. Do not use any external knowledge.

IMPORTANT: Base your answer STRICTLY on the information provided below. If the information doesn't fully answer the question, say so.

Information from knowledge base:
{context_text}

Question: {original_query}

Instructions:
- Answer only using the information provided above
- If information is incomplete, mention what's missing
- Do not add information from outside sources
- Provide a clear, direct answer based solely on the given context

Answer:"""

        try:
            print(f"🤖 SENDING PROMPT TO LLM:")
            print(f"📋 Context length: {len(context_text)} chars")
            
            response = self.llm_connector.generate_response(prompt)
            print(f"🤖 LLM GENERATED RESPONSE: '{response}'")
            
            # Fallback to dynamic summarization if LLM gives too short response
            if len(response.strip()) < 20:
                logger.warning("LLM response too short, using fallback summarization")
                from itertools import islice
                sentences = " ".join(context_chunks).split('. ')
                response = ". ".join(islice(sentences, 0, min(8, len(sentences)))) + "."
                print(f"📝 FALLBACK RESPONSE: '{response}'")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fallback summarization
            sentences = " ".join(context_chunks).split('. ')
            fallback_response = ". ".join(sentences[:5]) + "."
            print(f"🚨 ERROR FALLBACK: '{fallback_response}'")
            return fallback_response

    def _calculate_confidence(self, avg_score: float, query: str = "", 
                            context_chunks: List[str] = None) -> str:
        """
        Confidence calculation based on relevance score and keyword coverage.
        """
        base_confidence = "low"
        
        if avg_score > 0.6:
            base_confidence = "high"
        elif avg_score > 0.4:
            base_confidence = "medium"
        elif avg_score > 0.2:
            base_confidence = "low"
        else:
            base_confidence = "very_low"
        
        # Adjust confidence based on keyword presence
        if query and context_chunks:
            query_keywords = self._extract_keywords(query)
            context_text = " ".join(context_chunks).lower()
            
            keyword_matches = 0
            for kw in query_keywords:
                kw_lower = kw.lower()
                if " " in kw_lower:  # Multi-word phrase
                    if kw_lower in context_text:
                        keyword_matches += 1
                else:  # Single word
                    if kw_lower in context_text:
                        keyword_matches += 1
            
            keyword_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # Boost confidence if good keyword coverage
            if keyword_ratio >= 0.7:  # 70% of keywords found
                if base_confidence == "low":
                    base_confidence = "medium"
                elif base_confidence == "very_low":
                    base_confidence = "low"
        
        logger.info(f"📊 Confidence calculated: {base_confidence}")
        return base_confidence

    def _empty_response(self, query: str, msg: str) -> Dict[str, Any]:
        """Generate empty response structure."""
        return {
            'query': query,
            'answer': msg,
            'context_used': [],
            'sources': [],
            'confidence': 'low',
            'metadata': {
                'total_sources': 0,
                'avg_relevance_score': 0.0,
                'llm_provider': self.llm_connector.__class__.__name__,
            }
        }

    def _error_response(self, query: str, err: str) -> Dict[str, Any]:
        """Generate error response structure."""
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
            model_name = getattr(self.llm_connector, 'model', 'unknown')

            if is_available:
                # Test with a simple query
                test_response = self.llm_connector.generate_response("Hello")
                test_successful = len(test_response) > 0 and "error" not in test_response.lower()
                
                return {
                    'status': 'connected',
                    'provider': self.llm_connector.__class__.__name__,
                    'model' : model_name,
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