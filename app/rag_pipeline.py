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
        self._small_talk_clf = None  # lazy-init zero-shot classifier

    def process_query(self, query: str, top_k: int = 6, min_score: float = 0.2, session_id: Optional[str] = None, history_turns: int = 6) -> Dict[str, Any]:
        """
        Flow:
        1. Initial search with original query
        2. Filter by score
        3. Build context and generate answer
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
            
            # Step 1: Initial search (prefer acronym-only search if present)
            extracted = self._extract_keywords(query)
            acronyms_only = [k for k in extracted if k.isupper() and len(k) > 1]
            search_query = " ".join(acronyms_only) if acronyms_only else query
            if acronyms_only:
                logger.info(f"Using acronym-focused search query: '{search_query}'")
            # When using acronyms/phrases, widen the search a bit
            effective_top_k = max(top_k, 10) if acronyms_only else top_k
            initial_results = self.vector_store.search(search_query, top_k=effective_top_k)
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

            # Step 3.5: Load recent chat history for this session (if any)
            history_text = ""
            history_messages_count = 0
            if session_id and hasattr(self.vector_store, 'get_chat_history'):
                try:
                    rows = self.vector_store.get_chat_history(session_id)
                    # take the last N messages
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
                except Exception as e:
                    logger.warning(f"Failed to load chat history for session {session_id}: {e}")

            # Step 4: Prepare context and generate response
            context_chunks = self._prepare_context(search_results, query)
            sources = self._create_sources(search_results)

            logger.info("Generating response with LLM...")
            answer, use_context = self._generate_response(query, context_chunks, history_text)
            logger.info(f"LLM response generated: {len(answer)} characters")

            avg_score = sum(r['score'] for r in search_results) / len(search_results)
            confidence = self._calculate_confidence(avg_score, query, context_chunks)

            if not use_context:
                sources = []
                context_chunks=[]
                confidence = 'high'
                metadata= {
                    'total_sources': 0,
                    'avg_relevance_score': 0.0,
                    'llm_provider': self.llm_connector.__class__.__name__,
                    'bypass': 'small_talk'
                }
            else:
                metadata= {
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

        # Prefer acronyms if present: return ONLY acronyms and their lowercase variants
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        acronyms = [t for t in tokens if t.isupper() and len(t) > 1]
        if acronyms:
            dedup_acronyms = list(dict.fromkeys(acronyms))
            return dedup_acronyms + [a.lower() for a in dedup_acronyms]

        keywords = []

        # 1. Preserve acronyms (from original query, not lowercased) — robust to punctuation
        for w in tokens:
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

    def _prepare_context(self, results: List[Dict[str, Any]], query: str, max_chunks: int = 10, sentence_window: int = 1) -> List[str]:
        query_keywords = self._extract_keywords(query)
        print(f'🔍 Query keywords: {query_keywords}')

        ranked_contexts = []

        for result in results:
            text = result['text'].replace('--- Page', '').replace('---', '').strip()
            text = ' '.join(text.split())  # normalize whitespace
            # Better sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]\s*', text) if len(s.strip()) > 20]

            for idx, sentence in enumerate(sentences):
                score = 0
                sentence_lower = sentence.lower()

                for keyword in query_keywords:
                    keyword_lower = keyword.lower()
                    if " " in keyword_lower:
                        pattern = rf"{re.escape(keyword_lower)}"
                        if re.search(pattern, sentence_lower):
                            score += 3
                    else:
                        pattern = rf"\b{re.escape(keyword_lower)}\b"
                        if re.search(pattern, sentence_lower):
                            score += 2

                if score > 0:
                    start = max(0, idx - sentence_window)
                    end = min(len(sentences), idx + sentence_window + 1)
                    window_text = ". ".join(sentences[start:end])
                    ranked_contexts.append((score, window_text, result['score']))

        ranked_contexts.sort(key=lambda x: (x[0], x[2]), reverse=True)

        seen = set()
        top_sentences = []
        for score, sentence, res_score in ranked_contexts:
            if sentence not in seen:
                seen.add(sentence)
                top_sentences.append(sentence)
            if len(top_sentences) >= max_chunks:
                break

        # Force at least 5 sentences if available
        if len(top_sentences) < 5:
            for result in results:
                text = ' '.join(result['text'].split())
                sentences = [s.strip() for s in re.split(r'[.!?]\s*', text) if len(s.strip()) > 20]
                for s in sentences:
                    if s not in seen:
                        seen.add(s)
                        top_sentences.append(s)
                    if len(top_sentences) >= 5:
                        break
                if len(top_sentences) >= 5:
                    break

        print('top_sentences =====> ', top_sentences)
        print(f"📝 Selected {len(top_sentences)} sentences for context")

        return [". ".join(top_sentences) + "."] if top_sentences else []

    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create source information from search results - return only the most relevant source."""
        if not results:
            return []
        
        # Sort results by relevance score (highest first) and take only the top one
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_result = sorted_results[0]
        
        src = {
            'filename': top_result['filename'],
            'document_id': top_result['document_id'],
            'relevance_score': round(top_result['score'], 4),
            'metadata': top_result.get('metadata', {})  # Add this line
        }
        if top_result.get('chunk_id'):
            src['chunk_id'] = top_result['chunk_id']
        text = ' '.join(top_result['text'].replace('--- Page', '').replace('---', '').split())
        src['preview'] = text[:80] + "..." if len(text) > 80 else text
        
        return [src]

    def _generate_response(self, original_query: str, context_chunks: List[str], history_text: str = "") -> str:
        """Generate response strictly from provided context."""
        if not context_chunks and not history_text:
            return "I don't have information about this."

        # Combine all chunks with numbering for clarity
        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        history_block = ("Conversation history (use for context and follow-up resolution):\n" + history_text) if history_text else ""
        
        prompt = f"""
        You are a friendly and helpful assistant. 
        You have access to two sources of information:
        1. Conversation history → for casual flow, continuity, greetings, and follow-ups.
        2. Knowledge base(KB) → for factual grounding.

        Rules for answering:
        - If the user’s question is related to *context*, answer **only** using the knowledge base info.
            - Don't give answer outside the knowledge base or from own pretrained knowledge. don't use your own knowledge.
            - If you got from_context: False in the response so return simple answer like I don't have information about this.
            - If the Knowledge base contains relevant facts → answer naturally using them.
            - If not enough info in Knowledge base → respond politely and concisely that the info is missing, and guide them gently but only respond from knowledge base.
            - After every answer based on Knowledge base, add: `from_context: True`.IF not related to Knowledge base, add: `from_context: False`.
            - If your answer relies on anything outside Knowledge base → add: `from_context: False`.
        - For greetings, casual talk, or unrelated topics → answer naturally as a human-like assistant (short, friendly, warm). Do not mention “knowledge base” or “I don’t have data.”
        - Always keep answers concise, conversational, and human-sounding (avoid robotic phrasing).

        Conversation history:
        {history_block}

        Knowledge base:
        {context_text}

        User’s question:
        {original_query}

        Now provide the best possible answer following the above rules.
        Answer:
        """


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

            import re
            # Use regex to match 'from_context: False' as a whole word, case sensitive
            if re.search(r'\bfrom_context:\s*False\b', response):
                print('from_context: False')
                use_context = False
            else:
                print('from_context: True')
                use_context = True

            # Remove 'from_context: True' or 'from_context: False' from the response string, if present
            response = re.sub(r'\s*from_context:\s*(True|False)\s*', '', response, flags=re.IGNORECASE)
            return response.strip(), use_context
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            # Fallback summarization
            sentences = " ".join(context_chunks).split('. ')
            fallback_response = ". ".join(sentences[:5]) + "."
            print(f"🚨 ERROR FALLBACK: '{fallback_response}'")
            return fallback_response
    def _is_small_talk(self, query: str) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        # Prefer ML-based zero-shot classification to detect greetings/small talk
        try:
            if self._small_talk_clf is None:
                from transformers import pipeline
                # bart-large-mnli is widely used for zero-shot; downloads once and caches
                self._small_talk_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
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
            result = self._small_talk_clf(q, candidate_labels=candidate_labels, hypothesis_template="This text is {}.")
            labels = result.get('labels', [])
            scores = result.get('scores', [])
            if labels and scores:
                top_label = labels[0].lower()
                top_score = float(scores[0])
                greeting_like = {"greeting", "small talk", "chit-chat", "farewell", "well-being question", "social nicety", "casual conversation"}
                if top_label in greeting_like and top_score >= 0.70:
                    return True
                # Very short, non-informational utterances with high greeting/thanks/compliment score
                if len(q) <= 40 and top_label in greeting_like.union({"gratitude", "compliment"}) and top_score >= 0.60:
                    return True
            return False
        except Exception as _:
            # Fallback heuristic if transformers unavailable
            import re as _re
            ql = q.lower()
            if len(ql) <= 16 and _re.fullmatch(r"(hi+|hello+|hey+|yo|hiya|namaste|hola|bonjour|ciao)[!. ]*", ql):
                return True
            if _re.search(r"\b(how\s*(are|r)\s*(you|u)|how'?s\s*it\s*going|what'?s\s*up|how\s*have\s*you\s*been)\b", ql):
                return True
            return False

    def _generate_small_talk_response(self, user_text: str) -> str:
        prompt = f"""
You are a friendly assistant. The user sent a casual greeting or small talk:

User: {user_text}
"""
        try:
            reply = self.llm_connector.generate_response(prompt)
            return reply.strip() if isinstance(reply, str) and reply.strip() else "Hello! How can I help you today?"
        except Exception:
            return "Hello! How can I help you today?"

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
                    pattern = rf"\b{re.escape(kw_lower)}\b"
                    if re.search(pattern, context_text):
                        keyword_matches += 1
                else:  # Single word
                    pattern = rf"\b{re.escape(kw_lower)}\b"
                    if re.search(pattern, context_text):
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