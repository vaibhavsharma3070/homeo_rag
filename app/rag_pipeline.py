from typing import List, Dict, Any, Optional
from loguru import logger
from app.vector_store import PGVectorStore
from app.llm_connector import LLMFactory, OllamaConnector
from app.config import settings
from nltk.corpus import stopwords
from nltk import ngrams
import re
import numpy as np

from langchain.memory import ConversationBufferMemory
from langchain.embeddings import SentenceTransformerEmbeddings

STOPWORDS = set(stopwords.words("english"))

class RAGPipeline:
    """RAG pipeline with cost-efficient conversation memory using LangChain."""

    def __init__(self):
        if settings.vector_backend.lower() == 'pgvector':
            self.vector_store = PGVectorStore()
        self.llm_connector: OllamaConnector = LLMFactory.create_connector()

        # LangChain memory: stores conversation history
        self.embedding_model = SentenceTransformerEmbeddings(model_name=settings.embedding_model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # -------------------- MEMORY FUNCTIONS --------------------
    def add_to_memory(self, user_message: str, ai_response: str):
        """Add user and AI messages to memory."""
        self.memory.save_context({"input": user_message}, {"output": ai_response})

    def get_relevant_history(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k relevant conversation chunks using embeddings."""
        messages = self.memory.load_memory_variables({}).get("chat_history", [])
        if not messages:
            return []

        # Extract text from messages
        message_texts = [m.content for m in messages if hasattr(m, "content")]
        embeddings = self.embedding_model.embed_documents(message_texts)
        query_emb = self.embedding_model.embed_query(query)

        # Cosine similarity
        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored = [(msg, cosine(query_emb, emb)) for msg, emb in zip(messages, embeddings)]
        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        top_messages = [m.content for m, s in scored[:top_k]]
        return top_messages

    # -------------------- EXISTING FUNCTIONS --------------------
    def process_query(self, query: str, top_k: int = 6, min_score: float = 0.2) -> Dict[str, Any]:
        """
        Flow:
        1. Retrieve relevant conversation history
        2. Retrieve relevant KB
        3. Generate LLM answer using combined context
        """
        try:
            logger.info(f"Processing query: '{query}'")

            # Step 1: Retrieve relevant KB results
            extracted = self._extract_keywords(query)
            acronyms_only = [k for k in extracted if k.isupper() and len(k) > 1]
            search_query = " ".join(acronyms_only) if acronyms_only else query
            effective_top_k = max(top_k, 10) if acronyms_only else top_k
            kb_results = self.vector_store.search(search_query, top_k=effective_top_k)
            filtered_kb = [r for r in kb_results if r['score'] >= min_score]

            # Step 2: Retrieve relevant conversation history
            relevant_history = self.get_relevant_history(query, top_k=3)
            print('here is the relevant history =====', relevant_history)

            # Step 3: Combine KB + conversation
            context_chunks = [r['text'] for r in filtered_kb] + relevant_history
            if not context_chunks:
                return self._empty_response(query, f"I don't have information about '{query}'.")

            # Step 4: Generate answer using combined context
            answer = self._generate_response(query, context_chunks)

            # Step 5: Update memory
            self.add_to_memory(query, answer)

            # Step 6: Sources and confidence
            sources = self._create_sources(filtered_kb)
            avg_score = sum(r['score'] for r in filtered_kb) / len(filtered_kb) if filtered_kb else 0.0
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

    # -------------------- ORIGINAL HELPER FUNCTIONS --------------------
    def _extract_keywords(self, query: str) -> List[str]:
        query_clean = query.lower().replace("?", "").strip()
        words = [w for w in query_clean.split() if w not in STOPWORDS and len(w) > 2]
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        acronyms = [t for t in tokens if t.isupper() and len(t) > 1]
        if acronyms:
            dedup_acronyms = list(dict.fromkeys(acronyms))
            return dedup_acronyms + [a.lower() for a in dedup_acronyms]
        keywords = []
        for w in tokens:
            if w.isupper() and len(w) > 1:
                keywords.append(w)
        bigrams = [" ".join(bg) for bg in ngrams(words, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(words, 3)]
        meaningful_phrases = [p for p in trigrams + bigrams if p in query_clean]
        if meaningful_phrases:
            keywords.extend(meaningful_phrases)
        else:
            keywords.extend(words)
        return list(dict.fromkeys(keywords))

    def _generate_response(self, original_query: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "I don't have information about this."
        context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
        prompt = f"""
You are a helpful assistant. Answer ONLY based on the provided information below. Do not use any external knowledge.

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
            response = self.llm_connector.generate_response(prompt)
            if len(response.strip()) < 20:
                from itertools import islice
                sentences = " ".join(context_chunks).split('. ')
                response = ". ".join(islice(sentences, 0, min(8, len(sentences)))) + "."
            return response.strip()
        except Exception as e:
            sentences = " ".join(context_chunks).split('. ')
            fallback_response = ". ".join(sentences[:5]) + "."
            return fallback_response

    def _calculate_confidence(self, avg_score: float, query: str = "", context_chunks: List[str] = None) -> str:
        base_confidence = "low"
        if avg_score > 0.6:
            base_confidence = "high"
        elif avg_score > 0.4:
            base_confidence = "medium"
        elif avg_score > 0.2:
            base_confidence = "low"
        else:
            base_confidence = "very_low"
        if query and context_chunks:
            query_keywords = self._extract_keywords(query)
            context_text = " ".join(context_chunks).lower()
            keyword_matches = sum(
                1 for kw in query_keywords if re.search(rf"\b{re.escape(kw.lower())}\b", context_text)
            )
            keyword_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
            if keyword_ratio >= 0.7:
                if base_confidence == "low":
                    base_confidence = "medium"
                elif base_confidence == "very_low":
                    base_confidence = "low"
        return base_confidence

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
                'llm_provider': self.llm_connector.__class__.__name__,
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

    def _create_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
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