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
        min_score: float = 0.1, 
        session_id: Optional[str] = None, 
        history_turns: int = 4
    ) -> Dict[str, Any]:
        """Process query with agent-first approach, fallback to vector search."""
        try:
            logger.info(f"Processing query: '{query}'")

            # Handle small talk
            if self._is_small_talk(query):
                logger.info("Small talk detected - bypassing KB")
                answer = self._generate_small_talk_response(query)
                return self._create_response(query, answer, [], [], 'high', {'bypass': 'small_talk'})
            
            # Load history
            history_text = ""
            history_list = []
            if session_id and hasattr(self.vector_store, 'get_chat_history'):
                try:
                    rows = self.vector_store.get_chat_history(session_id)
                    recent = rows[-history_turns:] if len(rows) > history_turns else rows
                    history_text = "\n".join([f"{r['role'].upper()}: {r['message']}" for r in recent])
                    history_list = [{"role": r['role'], "message": r['message']} for r in recent]
                    logger.info(f"Loaded {len(recent)} history messages")
                except Exception as e:
                    logger.warning(f"Failed to load history: {e}")
            
            # STEP 1: TRY AGENT FIRST with history context
            logger.info("🤖 Step 1: Attempting intelligent agent search...")
            agent_result = None
            agent_failed = False
            
            try:
                agent_result = self.vector_store.search_with_agent(query, history=history_list)
                
                # Check if agent result is actually useful
                if agent_result:
                    agent_result_lower = agent_result.lower().strip()
                    
                    # Rejection phrases - agent couldn't find information
                    rejection_phrases = [
                        "couldn't find",
                        "don't have",
                        "do not have",
                        "don't know",
                        "no information",
                        "no results",
                        "not available",
                        "no relevant",
                        "no specific",
                        "no data",
                        "unable to find",
                        "cannot find"
                    ]
                    
                    # Check if ANY rejection phrase exists in the response
                    is_rejection = any(phrase in agent_result_lower for phrase in rejection_phrases)
                    
                    # Also check if result is too short (likely an error/rejection)
                    is_too_short = len(agent_result_lower) < 15
                    
                    if is_rejection or is_too_short:
                        logger.warning(f"⚠️ Agent returned insufficient result (rejection={is_rejection}, too_short={is_too_short})")
                        logger.warning(f"Agent response preview: '{agent_result[:100]}'")
                        agent_result = None  # Force fallback
                        agent_failed = True
                    else:
                        logger.info("✅ Agent successfully answered the query")
                        
                        return self._create_response(
                            query, 
                            agent_result, 
                            [agent_result[:500]], 
                            [{
                                'filename': 'Knowledge Base',
                                'document_id': 0,
                                'relevance_score': 0.95,
                                'preview': agent_result[:100] + "...",
                                'metadata': {'search_method': 'database_agent'}
                            }], 
                            'high', 
                            {
                                'total_sources': 1,
                                'avg_relevance_score': 0.95,
                                'search_method': 'agent',
                                'llm_provider': 'Gemini-Agent'
                            }
                        )
                else:
                    logger.warning("⚠️ Agent returned None/empty result")
                    agent_failed = True
                    
            except Exception as e:
                logger.error(f"❌ Agent search failed with exception: {e}")
                agent_result = None
                agent_failed = True
            
            # STEP 2: AGENT FAILED OR RETURNED INSUFFICIENT RESULT - FALLBACK TO VECTOR SEARCH
            logger.info("📊 Step 2: Agent couldn't answer - falling back to vector search...")
            
            try:
                search_results = self.vector_store.search(query, top_k=top_k)
                filtered = [r for r in search_results if r['score'] >= min_score]
                logger.info(f"✓ Vector search returned {len(filtered)} results (min_score={min_score})")

                if not filtered:
                    logger.warning("❌ No results from both agent and vector search")
                    return self._create_response(
                        query, 
                        "I don't have specific information about that in my knowledge base.",
                        [], [], 'low', {
                            'total_sources': 0, 
                            'search_method': 'none',
                            'agent_attempted': True,
                            'agent_failed': agent_failed,
                            'vector_attempted': True
                        }
                    )

                # STEP 3: GENERATE ANSWER FROM VECTOR SEARCH RESULTS
                logger.info("✓ Using vector search results to generate answer")
                context_chunks = self._prepare_context(filtered)
                sources = self._create_sources(filtered)
                answer = self._generate_response(query, context_chunks, history_text)

                avg_score = sum(r['score'] for r in filtered) / len(filtered)
                confidence = self._calculate_confidence(avg_score)
                metadata = {
                    'total_sources': len(sources),
                    'avg_relevance_score': round(avg_score, 4),
                    'search_method': 'vector_fallback',
                    'llm_provider': self.llm_connector.__class__.__name__,
                    'agent_attempted': True,
                    'agent_failed': agent_failed
                }

                return self._create_response(query, answer, context_chunks, sources, confidence, metadata)
                
            except Exception as vector_error:
                logger.error(f"❌ Vector search also failed: {vector_error}")
                return self._create_response(
                    query, 
                    "I encountered an error while searching the knowledge base. Please try again.",
                    [], [], 'error', {
                        'error': str(vector_error),
                        'agent_attempted': True,
                        'agent_failed': agent_failed,
                        'vector_attempted': True,
                        'vector_failed': True
                    }
                )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._create_response(
                query, 
                f"An error occurred: {str(e)}", 
                [], [], 'error', 
                {'error': str(e)}
            )

    def _prepare_context(self, results: List[Dict[str, Any]], max_chunks: int = 10) -> List[str]:
        """Extract unique context chunks from search results."""
        seen = set()
        chunks = []
        
        for r in results:
            text = ' '.join(r['text'].replace('--- Page', '').replace('---', '').split())
            print('text =========================================== ',text)
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
        system_parts = [
            "You are a helpful and knowledgeable assistant. Your task is to provide clear, accurate, and conversational answers based ONLY on the information provided in the Knowledge Base below.",
            "\nIMPORTANT RULES:",
            "- Synthesize information from the Knowledge Base into a natural, flowing response",
            "- Do NOT copy raw text directly - rephrase and summarize in your own words",
            "- Be conversational and easy to understand",
            "- If the question uses pronouns (he/she/it/they/this/that), check Recent Conversation to understand the reference",
            "- Do NOT mention 'Knowledge Base', 'records', 'documents', or 'conversation history' in your response",
            "- Do NOT say things like 'Based on the records' or 'I found X records'",
            "- If you cannot answer from the provided information, politely say you don't have that specific information",
            "- NEVER use external knowledge - only use what's provided below",
            "- Please don't add the markdown format in your response"
        ]
        
        if history:
            system_parts.append(f"\n## Recent Conversation:\n{history}")
        
        system_parts.append(f"\n## Knowledge Base:\n{context_block}")
        system_parts.append(f"\n## User Question:\n{query}")
        system_parts.append("\nProvide a clear, natural answer:")
        
        prompt = "\n".join(system_parts)
        
        try:
            logger.info(f"Sending prompt to LLM ({len(context_block)} chars context)")
            response = self.llm_connector.generate_response(prompt)
            logger.info(f"LLM response received ({len(response)} chars)")
            
            # Clean up response if it still contains unwanted phrases
            unwanted_phrases = [
                "based on the records",
                "according to the documents",
                "the records show",
                "from the knowledge base",
                "i found",
                "records related to"
            ]
            
            response_cleaned = response.strip()
            for phrase in unwanted_phrases:
                if phrase in response_cleaned.lower():
                    # Try to remove the phrase
                    response_cleaned = re.sub(
                        re.escape(phrase), 
                        "", 
                        response_cleaned, 
                        flags=re.IGNORECASE
                    ).strip()
            
            if len(response_cleaned) < 20:
                logger.warning("Response too short after cleaning, using original")
                response_cleaned = response.strip()
            
            if len(response_cleaned) < 20:
                logger.warning("Response still too short, generating fallback summary")
                # Create a better fallback summary
                first_chunk = context_chunks[0]
                sentences = first_chunk.split('. ')
                summary = '. '.join(sentences[:2]) + "."
                return summary
            
            return response_cleaned
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            # Better fallback
            first_chunk = context_chunks[0]
            sentences = first_chunk.split('. ')
            return '. '.join(sentences[:2]) + "."

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