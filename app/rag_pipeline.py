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

    def _resolve_pronoun_query(self, query: str, history_text: str) -> str:
        """
        Resolve pronouns in follow-up queries using conversation history.
        Returns the resolved query or original if no pronoun detected.
        """
        query_lower = query.lower().strip()
        
        # Check if query contains pronouns or follow-up indicators
        follow_up_patterns = [
            r'\b(him|her|it|them|his|hers|its|their|that|this|these|those)\b',
            r'\bmore (info|information|details|about)\b',
            r'\btell me more\b',
            r'\b(what|how) about (him|her|it|them|that|this)\b'
        ]
        
        has_pronoun = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
        
        if not has_pronoun or not history_text:
            return query
        
        # Extract the last topic from history
        try:
            # Look for the last substantial AI response
            lines = history_text.strip().split('\n')
            last_topic = None
            
            for line in reversed(lines):
                if line.startswith('USER:'):
                    user_msg = line.replace('USER:', '').strip()
                    # Extract subject from user question
                    words = user_msg.lower().split()
                    # Skip common words to find the subject
                    skip_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'you', 'mean', 'by', 'the', 'a', 'an'}
                    topic_words = [w.strip('?,.:;!') for w in words if w.lower() not in skip_words and len(w) > 2]
                    if topic_words:
                        last_topic = ' '.join(topic_words[:3])  # Take first few meaningful words
                        break
            
            if last_topic:
                # Create resolved query
                resolved = query_lower
                resolved = re.sub(r'\b(him|her|it|them|that|this)\b', last_topic, resolved)
                resolved = re.sub(r'\bmore (info|information|details) about\b', f'{last_topic}', resolved)
                logger.info(f"Resolved pronoun query: '{query}' -> '{resolved}'")
                return resolved
                
        except Exception as e:
            logger.warning(f"Failed to resolve pronoun: {e}")
        
        return query

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
        2. Load chat history
        3. Resolve pronouns in follow-up queries
        4. Search vector store using resolved query
        5. Filter by score
        6. Generate contextual response
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
            
            # Load recent chat history FIRST (before search)
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
                    history_messages_count = len(tail)
                    logger.info(f"Loaded {history_messages_count} chat history messages")
                except Exception as e:
                    logger.warning(f"Failed to load chat history for session {session_id}: {e}")
            
            # Resolve pronouns in query using history
            resolved_query = self._resolve_pronoun_query(query, history_text)
            
            # Search using resolved query
            search_results = self.vector_store.search(resolved_query, top_k=top_k)
            logger.info(f"LangChain search retrieved {len(search_results)} results for resolved query")

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

            # Prepare context from search results
            context_chunks = self._prepare_context(filtered_results)
            sources = self._create_sources(filtered_results)

            logger.info("Generating response with LLM...")
            answer = self._generate_response(query, context_chunks, history_text)
            logger.info(f"LLM response generated: {len(answer)} characters")

            # Calculate confidence
            avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
            confidence = self._calculate_confidence(avg_score)

            metadata = {
                'total_sources': len(sources),
                'avg_relevance_score': round(avg_score, 4),
                'llm_provider': self.llm_connector.__class__.__name__,
                'history_messages_used': history_messages_count,
                'query_resolved': resolved_query != query
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
    ) -> str:
        """Generate response from context and chat history, with pronoun resolution."""
        
        if not context_chunks and not history_text:
            return "I don't have information about this."

        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        
        history_block = (
            "Conversation history (for continuity and context):\n" + history_text
        ) if history_text else ""
        
        prompt = f"""You are a helpful assistant with access to a knowledge base and conversation history.

            {history_block}

            Knowledge Base:
            {context_text}

            User's Question: {original_query}

            Instructions:
            - If this is a follow-up question with pronouns (him/her/it/them/this/that), use conversation history to understand what the user is referring to, then answer using the Knowledge Base.
            - Answer using information from the Knowledge Base above.
            - If the Knowledge Base has relevant information, provide a natural, informative answer.
            - Be conversational and helpful.
            - Do not mention "Knowledge Base" or "context" in your response.

            Answer:
        """
        try:
            logger.info(f"🤖 Sending prompt to LLM (context: {len(context_text)} chars)")
            
            response = self.llm_connector.generate_response(prompt)
            print('response =========================================== ',response)
            logger.info(f"🤖 LLM response received: {len(response)} chars")
            
            # Fallback if response too short
            if len(response.strip()) < 20:
                logger.warning("LLM response too short, using fallback")
                sentences = " ".join(context_chunks).split('. ')
                response = ". ".join(sentences[:5]) + "."

            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            sentences = " ".join(context_chunks).split('. ')
            fallback_response = ". ".join(sentences[:3]) + "."
            return fallback_response

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

    # Utility methods remain the same...
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"Successfully added {len(documents)} documents to knowledge base")
            return success
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 10, min_score: float = 0.2) -> List[Dict[str, Any]]:
        try:
            results = self.vector_store.search(query, top_k=top_k)
            return [r for r in results if r['score'] >= min_score]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        try:
            stats = self.vector_store.get_index_stats()
            stats['llm_available'] = self.llm_connector.is_available()
            stats['llm_provider'] = self.llm_connector.__class__.__name__
            return stats
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {'error': str(e)}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.vector_store.get_all_documents()

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        return self.vector_store.get_document_by_id(doc_id)

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        return self.vector_store.get_document_chunks(doc_id)

    def clear_knowledge_base(self) -> bool:
        try:
            self.vector_store.clear_index()
            logger.info("Knowledge base cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False

    def test_llm_connection(self) -> Dict[str, Any]:
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