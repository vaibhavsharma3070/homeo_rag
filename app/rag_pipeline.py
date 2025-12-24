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

    def _resolve_pronoun_query(self, query: str, history_text: str, history_list: List[Dict] = None) -> str:
        """Resolve pronouns using conversation history with LLM-based entity extraction."""
        query_lower = query.lower().strip()
        
        # Detect pronouns and referential words
        pronoun_pattern = r'\b(he|she|him|her|his|hers|it|its|they|them|their|that|this|those|these|also|more|about)\b'
        
        if not re.search(pronoun_pattern, query_lower) or not history_text:
            return query
        
        try:
            # Use LLM to intelligently resolve pronouns by extracting entities from history
            resolve_prompt = f"""You are a query rewriter for a conversation system.

## Conversation History:
{history_text}

## Current User Question:
{query}

The current question contains pronouns or reference words (he/she/it/him/her/this/that/also/more/about).
Your task is to rewrite the question by replacing pronouns with the actual subject/entity from the conversation history.

IMPORTANT RULES:
1. Extract the main subject/entity (person name, thing, concept) from previous USER messages
2. Replace pronouns in the current question with the extracted entity
3. Keep the question natural and clear
4. Return ONLY the rewritten question, nothing else
5. If you cannot find a clear reference, return the original question

Examples:
History: "USER: Tell me about John Doe"
Question: "Give me more details about him"
Rewritten: "Give me more details about John Doe"

History: "USER: What is homeopathy?"
Question: "Tell me more about it"
Rewritten: "Tell me more about homeopathy"

History: "USER: Who is patient H001?"
Question: "What is his contact number?"
Rewritten: "What is the contact number of patient H001?"

Rewritten question:"""
            
            resolved_query = self.llm_connector.generate_response(resolve_prompt).strip()
            
            # Validate the resolved query
            if resolved_query and len(resolved_query) > len(query) * 0.5:  # Should be reasonable length
                logger.info(f"🔄 Pronoun resolved: '{query}' -> '{resolved_query}'")
                return resolved_query
            else:
                logger.warning(f"LLM resolution returned invalid result, using original query")
                return query
                
        except Exception as e:
            logger.warning(f"Pronoun resolution failed: {e}, using original query")
            return query

    def _is_factual_query(self, query: str) -> bool:
        """Dynamically detect if a query is asking for factual/informational data 
        (name, age, date, etc.) vs a consultation/symptom-based query using LLM."""
        if not query or not query.strip():
            return False
        
        # Quick check for identity questions - these should always get personalization
        query_lower = query.lower().strip()
        identity_patterns = [
            r'who are you',
            r'what is your name',
            r'what\'s your name',
            r'what are you',
            r'introduce yourself',
            r'tell me about yourself'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Identity question detected: '{query}' -> CONSULTATION (needs personalization)")
                return False  # Return False = CONSULTATION = apply personalization
        
        try:
            classification_prompt = f"""Analyze this user query and determine if it's asking for factual information or if it's a consultation/advice query.

Query: "{query}"

A FACTUAL query asks for specific data points from records/database like:
- Patient names, ages, dates, contact information
- Patient IDs, record numbers
- Specific values from records (height, weight, BMI, etc.)
- What was prescribed/recommended to a specific patient
- Historical information retrieval about patients

A CONSULTATION query asks for:
- Medical advice or recommendations
- Symptom analysis or treatment suggestions
- Lifestyle tips or guidance
- Explanations about remedies or conditions
- General health information
- Questions about yourself (who are you, what is your name, etc.)

IMPORTANT: Questions about the AI's identity ("who are you", "what is your name") are ALWAYS CONSULTATION queries, not FACTUAL.

Respond with ONLY one word: "FACTUAL" or "CONSULTATION"

Classification:"""
            
            response = self.llm_connector.generate_response(classification_prompt).strip().upper()
            
            # Check if response indicates factual query
            is_factual = "FACTUAL" in response and "CONSULTATION" not in response
            
            logger.info(f"Query classification: '{query}' -> {'FACTUAL' if is_factual else 'CONSULTATION'}")
            return is_factual
            
        except Exception as e:
            logger.warning(f"Failed to classify query dynamically: {e}, defaulting to consultation mode")
            # On error, default to consultation (apply personalization) to be safe
            return False

    def _apply_personalization_to_response(self, response: str, user_id: Optional[int] = None, query: Optional[str] = None) -> str:
        """Apply shared admin personalization to any response (applies to all users and admins).
        Personalization is shared across all admins.
        For factual queries, returns the response as-is without adding consultation advice."""
        
        # If this is a factual query, skip personalization and return direct answer
        if query and self._is_factual_query(query):
            logger.info(f"Factual query detected - skipping personalization to return direct answer")
            return response
        
        if not hasattr(self.vector_store, 'get_user_personalization') or not hasattr(self.vector_store, 'SessionLocal'):
            return response
        
        try:
            # Get first admin's personalization (shared across all admins, applies to all users)
            with self.vector_store.SessionLocal() as db:
                admin_user = db.query(self.vector_store.UserORM).filter_by(role='admin').order_by(self.vector_store.UserORM.id.asc()).first()
                if not admin_user:
                    return response
                personalization = self.vector_store.get_user_personalization(admin_user.id)
                if not personalization:
                    return response
            
            # Check if there's actually any personalization to apply (handle empty strings)
            has_custom_instructions = bool(personalization.get('custom_instructions', '').strip())
            has_nickname = bool(personalization.get('nickname', '').strip())
            tone = personalization.get('base_style_tone', 'default')
            has_custom_tone = tone and tone != 'default'
            
            # If no personalization settings exist, return original response
            if not (has_custom_instructions or has_nickname or has_custom_tone):
                logger.info(f"No admin personalization settings found, returning original response")
                return response
            
            # Build a refinement prompt only if we have something to personalize
            refinement_parts = []
            
            # Add AI identity from nickname first
            if has_nickname:
                nickname = personalization['nickname'].strip()
                refinement_parts.append(f"Your identity: You are {nickname}. This is your name. Always refer to yourself as {nickname}.")
            
            if has_custom_instructions:
                refinement_parts.append(f"User's custom instructions: {personalization['custom_instructions']}")
            
            if has_nickname:
                refinement_parts.append(f"User's preferred name: {personalization['nickname']}")
            
            if has_custom_tone:
                tone_map = {
                    'professional': 'Use a professional and formal tone.',
                    'casual': 'Use a casual and friendly tone.',
                    'friendly': 'Use a warm and approachable tone.',
                    'formal': 'Use a strictly formal and business-like tone.'
                }
                if tone in tone_map:
                    refinement_parts.append(tone_map[tone])
            
            if not refinement_parts:
                logger.info(f"No valid admin personalization rules, returning original response")
                return response
            
            # Check if response contains structured formatting (numbered lists, special sections)
            has_numbered_list = bool(re.search(r'\n\s*\d+\.', response))
            has_special_formatting = '**' in response or '##' in response or '\n\n' in response
            
            # If response has special formatting, preserve it and only adjust tone slightly
            if has_numbered_list or has_special_formatting:
                logger.info("Response has special formatting (numbered lists/sections) - preserving structure")
                
                # Only apply tone adjustments, not content rewriting
                if has_custom_tone:
                    tone_instruction = refinement_parts[-1] if has_custom_tone else ""
                    tone_prompt = f"""Adjust the tone of this response to be {tone_instruction.lower()}.
Keep ALL formatting, numbered lists, sections, and structure EXACTLY as-is.
Only change the conversational tone/style where appropriate.

Original response:
{response}

Adjusted response (preserve ALL formatting):"""
                    
                    logger.info(f"Applying tone adjustment only (preserving formatting)")
                    personalized = self.llm_connector.generate_response(tone_prompt).strip()
                    
                    # Validate formatting is preserved
                    if has_numbered_list and not re.search(r'\n\s*\d+\.', personalized):
                        logger.warning("Tone adjustment removed numbered list, using original")
                        return response
                    
                    return personalized if personalized else response
                else:
                    # No tone to apply, return original with formatting intact
                    return response
            
            # Build the full prompt
            full_prompt = f"""Original response: {response}

    User preferences:
    {chr(10).join(refinement_parts)}

    Task: Rewrite the response to match the user's preferences above. Keep all factual content exactly the same. Return ONLY the rewritten response with no explanations or meta-commentary.

    Rewritten response:"""
            
            logger.info(f"Applying admin personalization to response")
            personalized = self.llm_connector.generate_response(full_prompt).strip()
            
            # Validate the response isn't the prompt itself
            if "rewrite" in personalized.lower() or "original response" in personalized.lower():
                logger.warning("LLM returned the prompt instead of rewriting, using original response")
                return response
                
            return personalized if personalized else response
            
        except Exception as e:
            logger.warning(f"Could not apply personalization: {e}")
            return response

    def process_query(
        self, 
        query: str, 
        top_k: int = 5, 
        min_score: float = 0.1, 
        session_id: Optional[str] = None, 
        history_turns: int = 25,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process query with agent-first approach, fallback to vector search."""
        try:
            logger.info(f"Processing query: '{query}'")

            # Handle small talk
            if self._is_small_talk(query,user_id):
                logger.info("Small talk detected - bypassing KB")
                answer = self._generate_small_talk_response(query,user_id=user_id)
                return self._create_response(query, answer, [], [], 'high', {'bypass': 'small_talk'})
            
            # Load history (filtered by user_id so each user only sees their own chat history)
            history_text = ""
            history_list = []
            if session_id and hasattr(self.vector_store, 'get_chat_history'):
                try:
                    # Pass user_id to filter chat history per user (each admin has separate chat history)
                    rows = self.vector_store.get_chat_history(session_id, user_id=user_id)
                    recent = rows[-history_turns:] if len(rows) > history_turns else rows
                    history_text = "\n".join([f"{r['role'].upper()}: {r['message']}" for r in recent])
                    history_list = [{"role": r['role'], "message": r['message']} for r in recent]
                    logger.info(f"Loaded {len(recent)} history messages for user_id={user_id}")
                except Exception as e:
                    logger.warning(f"Failed to load history: {e}")
                    logger.warning(f"Could not get user_id: {e}")
            
            # Resolve pronouns in query using conversation history
            original_query = query
            resolved_query = query
            if history_text:
                resolved_query = self._resolve_pronoun_query(query, history_text, history_list)
                if resolved_query != original_query:
                    logger.info(f"✅ Query reformulated: '{original_query}' -> '{resolved_query}'")
                query = resolved_query  # Use resolved query for processing
            
            # STEP 1: TRY AGENT FIRST with history context
            logger.info("🤖 Step 1: Attempting intelligent agent search...")
            agent_result = None
            agent_filenames = []
            agent_failed = False
            
            try:
                agent_result, agent_filenames = self.vector_store.search_with_agent(query, history=history_list, user_id=user_id)
                
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
                    is_too_short = len(agent_result_lower) < 3
                    
                    if is_rejection or is_too_short:
                        logger.warning(f"⚠️ Agent returned insufficient result (rejection={is_rejection}, too_short={is_too_short})")
                        logger.warning(f"Agent response preview: '{agent_result[:100]}'")
                        agent_result = None  # Force fallback
                        agent_failed = True
                    
                    else:
                        logger.info("✅ Agent successfully answered the query")
                        
                        # ✅ GET ACTUAL SOURCES ONLY IF AGENT QUERIED THE KNOWLEDGE BASE (has filenames)
                        # Only show sources when agent actually retrieved data from KB, not for greetings/system queries
                        agent_sources = []
                        try:
                            if agent_filenames:
                                # Agent queried the KB and got filenames - search for documents matching those filenames
                                logger.info(f"🔍 Searching for sources matching agent filenames: {agent_filenames}")
                                source_results = self.vector_store.search_by_filenames(agent_filenames, query, top_k=5)
                                if source_results:
                                    agent_sources = self._create_sources(source_results)
                                    logger.info(f"✓ Fetched {len(agent_sources)} source(s) matching agent filenames")
                                else:
                                    # Filename search failed - but we know agent queried KB, so try regular search
                                    logger.warning("Filename search returned no results, trying regular vector search")
                                    source_results = self.vector_store.search(query, top_k=3)
                                    if source_results:
                                        agent_sources = self._create_sources(source_results)
                            else:
                                # No filenames = agent didn't query KB (greetings, system questions, etc.)
                                # Don't show sources for these types of queries
                                logger.info("No filenames from agent - this query didn't retrieve from KB, so no sources will be shown")
                        except Exception as src_error:
                            logger.warning(f"Could not fetch sources for agent result: {src_error}")
                        
                        print(f"🔍 DEBUG before personalized_answer: agent_result={agent_result}, user_id={user_id}")
                        personalized_answer = self._apply_personalization_to_response(
                            agent_result, 
                            user_id=user_id,
                            query=original_query
                        )
                        
                        # ✅ USE REAL SOURCES (with actual filenames)
                        context_preview = [agent_result[:500]] if agent_result else []
                        
                        return self._create_response(
                            original_query,
                            personalized_answer, 
                            context_preview, 
                            agent_sources,  # ✅ Real filenames from vector search
                            'high', 
                            {
                                'total_sources': len(agent_sources),
                                'avg_relevance_score': 0.95,
                                'search_method': 'agent',
                                'llm_provider': 'Gemini-Agent',
                                'original_query': original_query,
                                'resolved_query': resolved_query if resolved_query != original_query else None
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
                answer = self._generate_response(query, context_chunks, history_text, user_id=user_id)

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

                return self._create_response(original_query, answer, context_chunks, sources, confidence, {
                    **metadata,
                    'original_query': original_query,
                    'resolved_query': resolved_query if resolved_query != original_query else None
                })
                
            except Exception as vector_error:
                logger.error(f"❌ Vector search also failed: {vector_error}")
                return self._create_response(
                    query, 
                    "Currently I don't have the knowledge to answer this question. Please try again with a different question.",
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

    def _generate_response(self, query: str, context_chunks: List[str], history: str = "", user_id: Optional[int] = None) -> str:
        """Generate LLM response with context, history, and user personalization."""
        
        if not context_chunks and not history:
            return "I don't have information about this."

        context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(context_chunks)])
        
        # Build system prompt
        system_parts = ["You are a knowledgeable assistant. Answer questions using the provided information."]
        
        # Add admin's personalization (applies to all users)
        if hasattr(self.vector_store, 'get_user_personalization') and hasattr(self.vector_store, 'SessionLocal'):
            try:
                # Get admin user's personalization (applies to all users)
                with self.vector_store.SessionLocal() as db:
                    admin_user = db.query(self.vector_store.UserORM).filter_by(role='admin').first()
                    if admin_user:
                        personalization = self.vector_store.get_user_personalization(admin_user.id)
                        if personalization:
                            print(f"📥 Admin personalization applied for all users: {personalization}")
                            # Add custom instructions
                            if personalization.get('custom_instructions'):
                                print('added custom instructions')
                                system_parts.append(f"\n## Custom Instructions:\n{personalization['custom_instructions']}")
                            
                            # Add AI identity from nickname
                            if personalization.get('nickname'):
                                nickname = personalization['nickname'].strip()
                                print(f'added nickname as AI identity: {nickname}')
                                system_parts.append(f"\n## Your Identity:\nYou are {nickname}. This is your name. Always refer to yourself as {nickname}.")
                            
                            # Add user context
                            user_context_parts = []
                            if personalization.get('nickname'):
                                print('added nickname')
                                user_context_parts.append(f"User's preferred name: {personalization['nickname']}")
                            if personalization.get('occupation'):
                                print('added occupation')
                                user_context_parts.append(f"User's occupation: {personalization['occupation']}")
                            if personalization.get('more_about_you'):
                                print('added more about you')
                                user_context_parts.append(f"About user: {personalization['more_about_you']}")
                            
                            if user_context_parts:
                                system_parts.append(f"\n## User Context:\n" + "\n".join(user_context_parts))
                            
                            # Add style/tone preferences
                            tone = personalization.get('base_style_tone', 'default')
                            if tone != 'default':
                                tone_instructions = {
                                    'professional': 'Use a professional and formal tone.',
                                    'casual': 'Use a casual and friendly tone.',
                                    'friendly': 'Use a warm and approachable tone.',
                                    'formal': 'Use a strictly formal and business-like tone.'
                                }
                                if tone in tone_instructions:
                                    system_parts.append(f"\n## Response Style:\n{tone_instructions[tone]}")
            except Exception as e:
                logger.warning(f"Could not load personalization: {e}")
        
        if history:
            system_parts.append(f"\n## Recent Conversation:\n{history}")
        
        system_parts.append(f"\n## Knowledge Base:\n{context_block}")
        system_parts.append(f"\n## User Question:\n{query}")
        system_parts.append(f"""
    ## Instructions:
    - If the question uses pronouns (he/she/it/they/this/that), check Recent Conversation to understand the reference
    - Answer based on the Knowledge Base information
    - Be conversational and natural
    - Follow any custom instructions provided by the user
    - Use the user's preferred name if provided
    - Don't mention "Knowledge Base" or "conversation history" in your response
    - ONLY use information from the Knowledge Base and Recent Conversation above
    - Do NOT use external knowledge or make assumptions
    - Please don't add any markdown formatting in the response.
    - Do NOT use multiple languages in the response, Answer must be only in any one language.
    - Please Don't use Here's the rewritten response, adhering to the user's preferences while maintaining the factual content:, From database these kind of words in the response.
    - Please don't say Good morning each time
    
    
    ## RULES:
    - Please Don't be hallucinating, if you don't have the information, say so.
    - Please Don't be too verbose, be concise and to the point.

    Answer:""")
        
        prompt = "\n".join(system_parts)
        
        try:
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

            
    def _is_small_talk(self, query: str, user_id: Optional[int] = None) -> bool:
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

    def _generate_small_talk_response(self, text: str, user_id: Optional[int] = None) -> str:
        """Generate friendly response for greetings."""
        prompt = f"""Respond warmly to this greeting in 1-2 sentences:

User: {text}

Response:"""
        
        try:
            response = self.llm_connector.generate_response(prompt).strip()
            personalized_answer = self._apply_personalization_to_response(
                response, 
                user_id=user_id,
                query=text
            )
            return personalized_answer
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

    def delete_document(self, filename: str) -> bool:
        """Delete a document and all its chunks from the knowledge base."""
        try:
            return self.vector_store.delete_document(filename)
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

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