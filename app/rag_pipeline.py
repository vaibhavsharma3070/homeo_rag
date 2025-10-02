from typing import List, Dict, Any, Optional
from loguru import logger
from app.vector_store import PGVectorStore
from app.llm_connector import LLMFactory, GeminiConnector
from app.config import settings
import re


class RAGPipeline:
    """RAG pipeline with proper conversation context handling."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: GeminiConnector = LLMFactory.create_connector()
        self._small_talk_clf = None

    def _resolve_pronoun_query(self, query: str, history_text: str) -> str:
        """Resolve pronouns using conversation history."""
        query_lower = query.lower().strip()
        
        # Detect pronouns and referential words
        pronoun_pattern = r'\b(he|she|him|her|his|hers|it|its|they|them|their|that|this|those|these)\b'
        
        if not re.search(pronoun_pattern, query_lower) or not history_text:
            return query
        
        try:
            # Extract the main subject from the most recent USER question
            lines = history_text.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('USER:'):
                    user_msg = line.replace('USER:', '').strip().lower()
                    
                    # Extract subject using simple heuristics
                    # Look for "who is X" pattern
                    who_match = re.search(r'who\s+is\s+(?:the\s+)?(.+?)(?:\?|$)', user_msg)
                    if who_match:
                        subject = who_match.group(1).strip()
                        resolved = re.sub(pronoun_pattern, subject, query_lower)
                        logger.info(f"Resolved: '{query}' -> '{resolved}'")
                        return resolved
                    
                    # Fallback: take meaningful words after common question words
                    words = user_msg.split()
                    skip = {'who', 'what', 'where', 'when', 'is', 'are', 'the', 'a', 'an', 'current'}
                    subject_words = [w.strip('?,.:;!') for w in words if w not in skip and len(w) > 2]
                    
                    if subject_words:
                        subject = ' '.join(subject_words[:3])
                        resolved = re.sub(pronoun_pattern, subject, query_lower)
                        logger.info(f"Resolved: '{query}' -> '{resolved}'")
                        return resolved
                    break
                    
        except Exception as e:
            logger.warning(f"Pronoun resolution failed: {e}")
        
        return query

    def process_query(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.2, 
        session_id: Optional[str] = None, 
        history_turns: int = 4
    ) -> Dict[str, Any]:
        """
        Process query with context-aware search.
        
        Flow:
        1. Detect small talk -> bypass KB search
        2. Load chat history FIRST
        3. Resolve pronouns using history
        4. Search KB with resolved query
        5. Generate response with full context
        """
        try:
            logger.info(f"Processing query: '{query}'")

            # Handle small talk without KB search
            if self._is_small_talk(query):
                logger.info("Small talk detected - bypassing KB")
                answer = self._generate_small_talk_response(query)
                return self._create_response(query, answer, [], [], 'high', {'bypass': 'small_talk'})
            
            # LOAD HISTORY FIRST
            history_text = ""
            if session_id and hasattr(self.vector_store, 'get_chat_history'):
                try:
                    rows = self.vector_store.get_chat_history(session_id)
                    recent = rows[-history_turns:] if len(rows) > history_turns else rows
                    history_text = "\n".join([f"{r['role'].upper()}: {r['message']}" for r in recent])
                    logger.info(f"Loaded {len(recent)} history messages")
                except Exception as e:
                    logger.warning(f"Failed to load history: {e}")
            
            # Resolve pronouns AFTER loading history
            resolved_query = self.reform_query(query, history_text)
            print('resolved_query =========================================== ',resolved_query)

            # Search with resolved query
            search_results = self.vector_store.search(resolved_query, top_k=top_k)
            logger.info(f"Search returned {len(search_results)} results")

            # Filter by score
            filtered = [r for r in search_results if r['score'] >= min_score]
            logger.info(f"After filtering (min_score={min_score}): {len(filtered)} results")

            if not filtered:
                logger.info("No relevant results found")
                return self._create_response(
                    query, 
                    "I don't have specific information about that. Could you rephrase your question?",
                    [], [], 'low', {'total_sources': 0}
                )

            # Prepare context and sources
            context_chunks = self._prepare_context(filtered)
            sources = self._create_sources(filtered)

            # Generate response
            answer = self._generate_response(query, context_chunks, history_text)
            logger.info(f"Generated response: {len(answer)} chars")

            # Calculate metadata
            avg_score = sum(r['score'] for r in filtered) / len(filtered)
            confidence = self._calculate_confidence(avg_score)
            metadata = {
                'total_sources': len(sources),
                'avg_relevance_score': round(avg_score, 4),
                'llm_provider': self.llm_connector.__class__.__name__,
                'query_resolved': resolved_query != query
            }

            return self._create_response(query, answer, context_chunks, sources, confidence, metadata)

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._create_response(query, f"An error occurred: {str(e)}", [], [], 'error', {'error': str(e)})

    def _prepare_context(self, results: List[Dict[str, Any]], max_chunks: int = 3) -> List[str]:
        """Extract unique context chunks from search results."""
        seen = set()
        chunks = []
        
        for r in results:
            text = ' '.join(r['text'].replace('--- Page', '').replace('---', '').split())
            if len(text) < 20:
                continue
            
            fingerprint = text[:100].lower()
            if fingerprint not in seen:
                chunks.append(text)
                seen.add(fingerprint)
                if len(chunks) >= max_chunks:
                    break
        
        logger.info(f"Selected {len(chunks)} unique chunks from {len(results)} results")
        return chunks

    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create source metadata - returns only top source."""
        if not results:
            return []
        
        top = max(results, key=lambda x: x['score'])
        text = ' '.join(top['text'].replace('---', '').split())
        
        return [{
            'filename': top['filename'],
            'document_id': top['document_id'],
            'relevance_score': round(top['score'], 4),
            'preview': text[:80] + "..." if len(text) > 80 else text,
            'metadata': top.get('metadata', {})
        }]

    def _generate_response(self, query: str, context_chunks: List[str], history: str = "") -> str:
        """Generate LLM response with context and history."""
        
        if not context_chunks and not history:
            return "I don't have information about this."

        context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])
        
        # Build system prompt

        system_parts = ["You are a knowledgeable assistant. Answer questions using the provided information."]
        
        if history:
            system_parts.append(f"\n## Recent Conversation:\n{history}")
        
        system_parts.append(f"\n## Knowledge Base:\n{context_block}")
        system_parts.append(f"\n## User Question:\n{query}")
        system_parts.append("""
## Instructions:
- If the question uses pronouns (he/she/it/they/this/that), check Recent Conversation to understand the reference
- Answer based on the Knowledge Base information
- Be conversational and natural
- Don't mention "Knowledge Base" or "conversation history" in your response
- ONLY use information from the Knowledge Base and Recent Conversation above
- Do NOT use external knowledge or make assumptions
- If user will gave you the their name you can use their name in the response as for example: "Hello John, how can I help you today?"

Answer:""")
        
        prompt = "\n".join(system_parts)
        
        try:
            print('prompt =========================================== ',prompt)
            logger.info(f"Sending prompt to LLM ({len(context_block)} chars context)")
            response = self.llm_connector.generate_response(prompt)
            logger.info(f"LLM response received ({len(response)} chars)")
            
            if len(response.strip()) < 20:
                logger.warning("Response too short, using fallback")
                return ". ".join(context_chunks[0].split('. ')[:3]) + "."
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ". ".join(context_chunks[0].split('. ')[:3]) + "."

    def reform_query(self, query: str, history: str = "") -> str:
        """Generate LLM response with context and history."""
        
        system_parts = []
        if history:
            system_parts.append(f"\n## Recent Conversation:\n{history}")
        
        system_parts.append(f"\n## User Question:\n{query}")
        system_parts.append("""
        You are a query rewriter for a Retrieval-Augmented Generation system.

        Your task:
        - Rewrite the latest user question into a self-contained, explicit query.
        - Use the conversation history to resolve pronouns or vague references.
        - Do not add any extra information that is not present in the conversation.
        - Keep the rewritten query clear and concise.
        - Return ONLY the rewritten query, nothing else.

        Conversation history:
        {history}

        Latest user query:
        {query}

        Rewritten query:

        Answer:""")
        
        prompt = "\n".join(system_parts)
        
        try:
            print('prompt for query formatting =========================================== ',prompt)
            response = self.llm_connector.generate_response(prompt)
            logger.info(f"LLM response received ({len(response)} chars)")
            
            return response.strip() 
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return query

            
    def _is_small_talk(self, query: str) -> bool:
        """Detect greetings and small talk."""
        q = query.strip().lower()
        if len(q) == 0:
            return False
        
        # Quick pattern matching
        greetings = ['hi', 'hello', 'hey', 'yo', 'hiya', 'namaste', 'hola']
        if len(q) <= 16 and any(q.startswith(g) for g in greetings):
            return True
        
        if re.search(r'\b(how\s*(are|r)\s*(you|u)|how\'?s\s*it|what\'?s\s*up)\b', q):
            return True
        
        return False

    def _generate_small_talk_response(self, text: str) -> str:
        """Generate friendly response for greetings."""
        prompt = f"""Respond warmly to this greeting in 1-2 sentences:

User: {text}

Response:"""
        
        try:
            return self.llm_connector.generate_response(prompt).strip()
        except:
            return "Hello! How can I help you today?"

    def _calculate_confidence(self, score: float) -> str:
        """Map relevance score to confidence level."""
        if score > 0.6:
            return "high"
        elif score > 0.4:
            return "medium"
        elif score > 0.2:
            return "low"
        return "very_low"

    def _create_response(
        self, 
        query: str, 
        answer: str, 
        context: List[str], 
        sources: List[Dict], 
        confidence: str, 
        metadata: Dict
    ) -> Dict[str, Any]:
        """Create standardized response structure."""
        metadata['total_sources'] = len(sources)
        metadata['avg_relevance_score'] = metadata.get('avg_relevance_score', 0.0)
        metadata['llm_provider'] = self.llm_connector.__class__.__name__
        
        return {
            'query': query,
            'answer': answer,
            'context_used': context,
            'sources': sources,
            'confidence': confidence,
            'metadata': metadata
        }

    # Utility methods
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            success = self.vector_store.add_documents(documents)
            if success:
                logger.info(f"Added {len(documents)} documents")
            return success
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 10, min_score: float = 0.2) -> List[Dict[str, Any]]:
        try:
            results = self.vector_store.search(query, top_k=top_k)
            return [r for r in results if r['score'] >= min_score]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        try:
            stats = self.vector_store.get_index_stats()
            stats['llm_available'] = self.llm_connector.is_available()
            stats['llm_provider'] = self.llm_connector.__class__.__name__
            return stats
        except Exception as e:
            logger.error(f"Stats error: {e}")
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
            logger.info("Knowledge base cleared")
            return True
        except Exception as e:
            logger.error(f"Clear error: {e}")
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