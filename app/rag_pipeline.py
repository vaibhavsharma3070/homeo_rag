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
    """Enhanced RAG pipeline with query expansion fallback for better retrieval."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: OllamaConnector = LLMFactory.create_connector()
        logger.info("Enhanced RAG Pipeline initialized with query expansion")

    def process_query(self, query: str, top_k: int = 3, min_score: float = 0.2) -> Dict[str, Any]:
        """
        Main query processing with fallback to query expansion.
        Flow:
        1. Initial search with original query
        2. If low/no matches → expand query with LLM
        3. Search again with expanded query
        4. Return summarized answer
        """
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Step 1: Initial search with original query
            initial_results = self.vector_store.search(query, top_k=top_k)
            logger.info(f"Initial search retrieved {len(initial_results)} results")

            # Step 2: Check if we need query expansion (low/no matches)
            filtered_initial = [r for r in initial_results if r['score'] >= min_score]
            print(f"filtered_initial :- {filtered_initial[:2] if len(filtered_initial) > 2 else filtered_initial}")  # Show first 2 for brevity
            logger.info(f"After filtering: {len(filtered_initial)} results remain")

            search_results = filtered_initial
            used_expanded_query = False
            expanded_query = query

            # Step 3: Check if expansion should be triggered
            should_expand = self._should_expand_query(filtered_initial, min_score, query)
            logger.info(f"Should expand query '{query}'? {should_expand}")
            
            if should_expand:
                logger.info("🔄 TRIGGERING QUERY EXPANSION - Low/no matches found")
                expanded_query = self._expand_query_with_llm(query)
                
                if expanded_query and expanded_query.lower() != query.lower():
                    logger.info(f"✅ Query expanded: '{query}' → '{expanded_query}'")
                    print(f"🔍 EXPANDED QUERY: '{expanded_query}'")
                    
                    expanded_results = self.vector_store.search(expanded_query, top_k=top_k)
                    filtered_expanded = [r for r in expanded_results if r['score'] >= min_score]
                    # ADD THIS PRINT for expanded results
                    print(f"filtered_expanded :- {filtered_expanded[:2] if len(filtered_expanded) > 2 else filtered_expanded}")
                    logger.info(f"Expanded search retrieved {len(filtered_expanded)} filtered results")
                    print(f"expanded_results count: {len(filtered_expanded)}")
                    
                    # Better logic for choosing between original and expanded results
                    expanded_better = False
                    if len(filtered_expanded) > len(filtered_initial):
                        expanded_better = True
                    elif len(filtered_expanded) == len(filtered_initial) and filtered_expanded:
                        # Compare average scores if same number of results
                        avg_expanded = sum(r['score'] for r in filtered_expanded) / len(filtered_expanded)
                        avg_initial = sum(r['score'] for r in filtered_initial) / len(filtered_initial) if filtered_initial else 0
                        if avg_expanded > avg_initial:
                            expanded_better = True
                    
                    if expanded_better:
                        search_results = filtered_expanded
                        used_expanded_query = True
                        logger.info("✅ Using expanded query results (better matches found)")
                    else:
                        logger.info("❌ Expanded query didn't improve results, using original")
                        # Use original results and original query for context
                        search_results = filtered_initial
                        expanded_query = query
                else:
                    logger.info("❌ Query expansion failed or returned same query")
            else:
                logger.info("✅ Direct search successful - No expansion needed")

            # Step 4: Handle no results case
            if not search_results:
                logger.info("❌ No results found even after expansion attempt")
                return self._empty_response(
                    query, 
                    f"I don't have information about '{query}'. Please try rephrasing your question or provide more context."
                )

            # Step 5: Prepare context and generate response
            context_chunks = self._prepare_context(search_results, expanded_query if used_expanded_query else query)
            sources = self._create_sources(search_results)

            logger.info("🤖 Generating response with LLM...")
            answer = self._generate_response(query, context_chunks, used_expanded_query, expanded_query)
            logger.info(f"✅ LLM response generated: {len(answer)} characters")

            avg_score = sum(r['score'] for r in search_results) / len(search_results)
            confidence = self._calculate_confidence(avg_score, query, context_chunks, used_expanded_query)

            return {
                'query': query,
                'expanded_query': expanded_query if used_expanded_query else None,
                'answer': answer,
                'context_used': context_chunks,
                'sources': sources,
                'confidence': confidence,
                'metadata': {
                    'total_sources': len(sources),
                    'avg_relevance_score': round(avg_score, 4),
                    'llm_provider': self.llm_connector.__class__.__name__,
                    'used_query_expansion': used_expanded_query,
                    'fallback_triggered': used_expanded_query,
                    'expansion_decision': {
                        'should_expand': should_expand,
                        'original_results_count': len(filtered_initial),
                        'expanded_results_count': len(search_results) if used_expanded_query else None
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_response(query, str(e))

    def _should_expand_query(self, results: List[Dict[str, Any]], min_score: float, query: str) -> bool:
        """
        Determine if query expansion should be triggered.
        More aggressive expansion triggers:
        - No results above min_score
        - Low average score even with results
        - Query appears to be an acronym or short form
        - Very few results (< 3)
        """
        # No results at all
        if not results:
            logger.info("🔄 Expansion trigger: No results found")
            return True
        
        # Very few results
        if len(results) < 3:
            logger.info(f"🔄 Expansion trigger: Too few results ({len(results)} < 3)")
            return True
        
        # Low average score
        avg_score = sum(r['score'] for r in results) / len(results)
        if avg_score < (min_score + 0.15):  # More sensitive threshold
            logger.info(f"🔄 Expansion trigger: Low average score ({avg_score:.3f} < {min_score + 0.15:.3f})")
            return True
        
        # Query appears to be acronym (2-4 uppercase letters)
        query_clean = query.strip().replace("?", "").replace("what is", "").replace("what", "").strip()
        if len(query_clean) <= 4 and query_clean.isupper():
            logger.info(f"🔄 Expansion trigger: Looks like acronym '{query_clean}'")
            return True
            
        # All results have relatively low scores
        max_score = max(r['score'] for r in results)
        if max_score < 0.4:  # No high-confidence results
            logger.info(f"🔄 Expansion trigger: Max score too low ({max_score:.3f} < 0.4)")
            return True
        
        logger.info(f"✅ No expansion needed: {len(results)} results, avg_score={avg_score:.3f}, max_score={max_score:.3f}")
        return False

    def _expand_query_with_llm(self, query: str) -> str:
        """
        Use LLM to expand/clarify the query.
        Handles acronyms, abbreviations, and ambiguous terms.
        """
        if not self.llm_connector.is_available():
            logger.warning("LLM not available for query expansion")
            return query

        expansion_prompt = f"""
        You are a helpful assistant that expands and clarifies search queries to improve information retrieval.

        Given the query: "{query}"

        If this appears to be:
        1. An acronym (like "IT", "AI", "API", "ML", "NLP") - provide the full form
        2. An abbreviation - provide the complete term  
        3. A technical term - add related keywords
        4. Already clear and specific - return it as is

        Respond with ONLY the expanded/clarified version. Do not add explanations.

        Examples:
        - "IT" → "Information Technology"
        - "AI" → "Artificial Intelligence"
        - "API" → "Application Programming Interface"
        - "ML" → "Machine Learning"
        - "what is python" → "what is python programming language"
        - "NLP" → "Natural Language Processing"

        Query: {query}
        Expanded query:"""

        try:
            logger.info(f"🤖 Requesting LLM expansion for: '{query}'")
            expanded = self.llm_connector.generate_response(expansion_prompt)
            print(f"🤖 LLM EXPANSION OUTPUT: '{expanded}'")
            
            cleaned_expansion = self._clean_expanded_query(expanded, query)
            
            if cleaned_expansion and len(cleaned_expansion) > len(query):
                logger.info(f"✅ Successfully expanded: '{query}' → '{cleaned_expansion}'")
                return cleaned_expansion
            else:
                logger.info(f"❌ Expansion didn't improve query: '{expanded}' → '{cleaned_expansion}'")
            
        except Exception as e:
            logger.error(f"Error expanding query with LLM: {e}")
        
        return query  # Return original if expansion fails

    def _clean_expanded_query(self, expanded: str, original: str) -> str:
        """Clean and validate the expanded keyword."""
        if not expanded:
            return original
        
        # Remove common prefixes/artifacts
        cleaned = expanded.strip()
        for prefix in ["Query:", "Expanded query:", "Answer:", "Result:", "Response:", "Full form:"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip(' :').strip()
        
        # Remove quotes if they wrap the entire query
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()
        
        # Ensure it's reasonable expansion (not too long)
        if len(cleaned) > len(original) * 8:  # Max 8x expansion for individual keywords
            logger.warning(f"Keyword expansion too long, using original")
            return original
            
        return cleaned if cleaned else original

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

    def _generate_response(self, original_query: str, context_chunks: List[str], 
                          used_expansion: bool = False, expanded_query: str = "") -> str:
        """Generate response with context about query expansion if used."""
        if not context_chunks:
            return "I don't have information about this."

        # Combine all chunks with numbering for clarity
        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        
        # More strict prompt to ensure only context-based answers
        if used_expansion and expanded_query:
            prompt = f"""
You are a helpful assistant. Answer ONLY based on the provided information below. Do not use any external knowledge.

The user asked about "{original_query}" which was expanded to "{expanded_query}" to find relevant information.

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
        else:
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
                            context_chunks: List[str] = None, used_expansion: bool = False) -> str:
        """
        Enhanced confidence calculation considering query expansion.
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
        
        # Adjust confidence based on query expansion and keyword presence
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
            
            # Slight penalty for using query expansion (less precise)
            if used_expansion and base_confidence == "high":
                base_confidence = "medium"
        
        logger.info(f"📊 Confidence calculated: {base_confidence} (expansion_used: {used_expansion})")
        return base_confidence

    def _empty_response(self, query: str, msg: str) -> Dict[str, Any]:
        """Generate empty response structure."""
        return {
            'query': query,
            'expanded_query': None,
            'answer': msg,
            'context_used': [],
            'sources': [],
            'confidence': 'low',
            'metadata': {
                'total_sources': 0,
                'avg_relevance_score': 0.0,
                'llm_provider': self.llm_connector.__class__.__name__,
                'used_query_expansion': False,
                'fallback_triggered': False
            }
        }

    def _error_response(self, query: str, err: str) -> Dict[str, Any]:
        """Generate error response structure."""
        return {
            'query': query,
            'expanded_query': None,
            'answer': f"An error occurred while processing your query: {err}",
            'context_used': [],
            'sources': [],
            'confidence': 'error',
            'metadata': {
                'total_sources': 0,
                'avg_relevance_score': 0.0,
                'llm_provider': self.llm_connector.__class__.__name__,
                'used_query_expansion': False,
                'fallback_triggered': False,
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