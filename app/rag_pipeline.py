from typing import List, Dict, Any, Optional
from loguru import logger
from app.vector_store import PGVectorStore
from app.llm_connector import LLMFactory, GeminiConnector
from app.config import settings
import re


class RAGPipeline:
    """RAG pipeline optimized for LangChain similarity search."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: GeminiConnector = LLMFactory.create_connector()
        self._small_talk_clf = None  # lazy-init zero-shot classifier

    def process_query(
        self, 
        query: str, 
        top_k: int = 8, 
        min_score: float = 0.2, 
        session_id: Optional[str] = None, 
        history_turns: int = 6
    ) -> Dict[str, Any]:
        """
        Process query with LangChain similarity search.
        
        Flow:
        1. Check for small talk/greetings
        2. Search vector store using LangChain
        3. Filter by score
        4. Load chat history
        5. Generate contextual response
        """
        try:
            logger.info(f"Processing query: '{query}'")

            # Small talk / greeting bypass: do not search KB or show sources
            if self._is_small_talk(query):
                logger.info("Detected small talk/greeting. Bypassing KB search and sources.")
                answer = self._generate_small_talk_response(query)
                return {
                    'query': query,
                    'answer': answer,
                    'context_used': [],
                    'sources': [],
                    'confidence': 'high',
                    'metadata': {
                        'total_sources': 0,
                        'avg_relevance_score': 0.0,
                        'llm_provider': self.llm_connector.__class__.__name__,
                        'bypass': 'small_talk'
                    }
                }
            
            # Search using LangChain's similarity search
            search_results = self.vector_store.search(query, top_k=top_k)
            logger.info(f"LangChain search retrieved {len(search_results)} results")

            # Filter by minimum score
            filtered_results = [r for r in search_results if r['score'] >= min_score]
            logger.info(f"After filtering (min_score={min_score}): {len(filtered_results)} results remain")

            # Handle no results case
            if not filtered_results:
                logger.info("No results found for the given query")
                return self._empty_response(
                    query, 
                    f"I don't have information about '{query}'. Please try rephrasing your question or provide more context."
                )

            # Load recent chat history for this session (if any)
            history_text = ""
            history_messages_count = 0
            if session_id and hasattr(self.vector_store, 'get_chat_history'):
                try:
                    rows = self.vector_store.get_chat_history(session_id)
                    tail = rows[-history_turns:]
                    formatted = []
                    for r in tail:
                        role = r.get('role', 'user')
                        msg = r.get('message', '')
                        if not isinstance(msg, str):
                            msg = str(msg)
                        formatted.append(f"{role.upper()}: {msg.strip()}")
                    history_text = "\n".join(formatted)
                    print('history_text =========================================== ',history_text)
                    history_messages_count = len(tail)
                    logger.info(f"Loaded {history_messages_count} chat history messages")
                except Exception as e:
                    logger.warning(f"Failed to load chat history for session {session_id}: {e}")

            # Prepare context from search results
            context_chunks = self._prepare_context(filtered_results)
            sources = self._create_sources(filtered_results)

            logger.info("Generating response with LLM...")
            answer, use_context = self._generate_response(query, context_chunks, history_text)
            logger.info(f"LLM response generated: {len(answer)} characters")

            # Calculate confidence
            avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
            confidence = self._calculate_confidence(avg_score)

            # If LLM determined not to use context, clear sources
            if not use_context:
                sources = []
                context_chunks = []
                confidence = 'high'
                metadata = {
                    'total_sources': 0,
                    'avg_relevance_score': 0.0,
                    'llm_provider': self.llm_connector.__class__.__name__,
                    'bypass': 'no_context_needed'
                }
            else:
                metadata = {
                    'total_sources': len(sources),
                    'avg_relevance_score': round(avg_score, 4),
                    'llm_provider': self.llm_connector.__class__.__name__,
                    'history_messages_used': history_messages_count
                }

            return {
                'query': query,
                'answer': answer,
                'context_used': context_chunks,
                'sources': sources,
                'confidence': confidence,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_response(query, str(e))

    def _prepare_context(self, results: List[Dict[str, Any]], max_chunks: int = 3) -> List[str]:
        """
        Prepare context from LangChain search results with deduplication.
        Reduces max_chunks to 3 for more focused, accurate context.
        """
        context_chunks = []
        seen_texts = set()
        
        for result in results:
            text = result['text'].replace('--- Page', '').replace('---', '').strip()
            text = ' '.join(text.split())  # Normalize whitespace
            
            if not text or len(text) < 20:
                continue
            
            # Create a fingerprint for deduplication (first 100 chars)
            fingerprint = text[:100].lower()
            
            # Skip if we've seen very similar text
            is_duplicate = False
            for seen in seen_texts:
                if fingerprint in seen or seen in fingerprint:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                context_chunks.append(text)
                seen_texts.add(fingerprint)
                
                if len(context_chunks) >= max_chunks:
                    break
        
        logger.info(f"📝 Selected {len(context_chunks)} unique context chunks from {len(results)} results")
        return context_chunks

    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create source information - return only the most relevant source."""
        if not results:
            return []
        
        # Sort by relevance score and take the top result
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_result = sorted_results[0]
        
        src = {
            'filename': top_result['filename'],
            'document_id': top_result['document_id'],
            'relevance_score': round(top_result['score'], 4),
            'metadata': top_result.get('metadata', {})
        }
        
        if top_result.get('chunk_id'):
            src['chunk_id'] = top_result['chunk_id']
        
        text = ' '.join(top_result['text'].replace('--- Page', '').replace('---', '').split())
        src['preview'] = text[:80] + "..." if len(text) > 80 else text
        
        return [src]

    def _generate_response(
        self, 
        original_query: str, 
        context_chunks: List[str], 
        history_text: str = ""
    ) -> tuple[str, bool]:
        """Generate response from context and chat history, with pronoun resolution."""
        
        if not context_chunks and not history_text:
            return "I don't have information about this.", False

        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        
        history_block = (
            "Conversation history (use for context, follow-up resolution, and pronoun references):\n" + history_text
        ) if history_text else ""
        
        prompt = f"""
    You are a friendly and helpful assistant. 
    You have access to two sources of information:
    1. Conversation history → for continuity, follow-ups, and pronoun resolution.
    2. Knowledge base (KB) → for factual grounding.

    Rules for answering:
    - Use conversation history to resolve pronouns (e.g., "him", "her", "it") in follow-up questions.
    - Answer questions **only using the Knowledge Base** if the user's question is factual.
        - If relevant info exists in KB → answer naturally using it.
        - If info is missing in KB → respond politely that info is not available, and guide the user gently.
    - For follow-ups referring to previous answers (using pronouns), resolve the reference using conversation history before looking up KB.
    - Add `from_context: True` if your answer relies on KB. Add `from_context: False` if you answer outside KB or info is missing.
    - For greetings or casual talk → answer naturally and friendly, ignore KB.

    Conversation history:
    {history_block}

    Knowledge base:
    {context_text}

    User's question:
    {original_query}

    Now provide the best possible answer following the above rules.
    Answer:
    """
        try:
            print('prompt =========================================== ',prompt)
            logger.info(f"🤖 Sending prompt to LLM (context: {len(context_text)} chars)")
            
            response = self.llm_connector.generate_response(prompt)
            logger.info(f"🤖 LLM response received: {len(response)} chars")
            
            # Fallback if response too short
            if len(response.strip()) < 20:
                logger.warning("LLM response too short, using fallback")
                sentences = " ".join(context_chunks).split('. ')
                response = ". ".join(sentences[:5]) + "."

            # Check if context was used
            use_context = not re.search(r'\bfrom_context:\s*False\b', response)
            
            # Remove the from_context marker from response
            response = re.sub(r'\s*from_context:\s*(True|False)\s*', '', response, flags=re.IGNORECASE)
            
            logger.info(f"✓ Context used: {use_context}")
            return response.strip(), use_context
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            sentences = " ".join(context_chunks).split('. ')
            fallback_response = ". ".join(sentences[:3]) + "."
            return fallback_response, True

    def _is_small_talk(self, query: str) -> bool:
        """Detect small talk and greetings using zero-shot classification."""
        q = (query or "").strip()
        if not q:
            return False
        
        try:
            if self._small_talk_clf is None:
                from transformers import pipeline
                self._small_talk_clf = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli"
                )
            
            candidate_labels = [
                "greeting",
                "small talk",
                "chit-chat",
                "farewell",
                "gratitude",
                "compliment",
                "well-being question",
                "social nicety",
                "casual conversation",
                "question",
                "information request"
            ]
            
            result = self._small_talk_clf(
                q, 
                candidate_labels=candidate_labels, 
                hypothesis_template="This text is {}."
            )
            
            labels = result.get('labels', [])
            scores = result.get('scores', [])
            
            if labels and scores:
                top_label = labels[0].lower()
                top_score = float(scores[0])
                
                greeting_like = {
                    "greeting", "small talk", "chit-chat", "farewell", 
                    "well-being question", "social nicety", "casual conversation"
                }
                
                if top_label in greeting_like and top_score >= 0.70:
                    return True
                
                if len(q) <= 40 and top_label in greeting_like.union({"gratitude", "compliment"}) and top_score >= 0.60:
                    return True
            
            return False
            
        except Exception:
            # Fallback heuristic
            ql = q.lower()
            if len(ql) <= 16 and re.fullmatch(r"(hi+|hello+|hey+|yo|hiya|namaste|hola|bonjour|ciao)[!. ]*", ql):
                return True
            if re.search(r"\b(how\s*(are|r)\s*(you|u)|how'?s\s*it\s*going|what'?s\s*up|how\s*have\s*you\s*been)\b", ql):
                return True
            return False

    def _generate_small_talk_response(self, user_text: str) -> str:
        """Generate response for small talk using LLM."""
        prompt = f"""You are a friendly assistant. The user sent a casual greeting or small talk:

User: {user_text}

Respond warmly and naturally in 1-2 sentences."""
        
        try:
            reply = self.llm_connector.generate_response(prompt)
            return reply.strip() if isinstance(reply, str) and reply.strip() else "Hello! How can I help you today?"
        except Exception:
            return "Hello! How can I help you today?"

    def _calculate_confidence(self, avg_score: float) -> str:
        """Calculate confidence based on average relevance score."""
        if avg_score > 0.6:
            return "high"
        elif avg_score > 0.4:
            return "medium"
        elif avg_score > 0.2:
            return "low"
        else:
            return "very_low"

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

    # Utility methods for knowledge base management
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

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the knowledge base."""
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
            model_name = getattr(self.llm_connector, 'model_name', 'unknown')

            if is_available:
                test_response = self.llm_connector.generate_response("Hello")
                test_successful = len(test_response) > 0 and "error" not in test_response.lower()
                
                return {
                    'status': 'connected',
                    'provider': self.llm_connector.__class__.__name__,
                    'model': model_name,
                    'test_successful': test_successful,
                    'test_response': test_response[:100] + "..." if len(test_response) > 100 else test_response
                }
            else:
                return {
                    'status': 'disconnected',
                    'provider': self.llm_connector.__class__.__name__,
                    'model': model_name,
                    'error': 'LLM service not available'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'provider': self.llm_connector.__class__.__name__,
                'model': 'unknown',
                'error': str(e)
            }