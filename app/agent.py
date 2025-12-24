# agent.py

import os
import sys
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import psycopg2
from psycopg2.extras import RealDictCursor

# Setup project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from app.config import settings

# Database configuration
DB_PARAMS = {
    "host": "localhost",
    "database": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
    "port": settings.postgres_port,
}

TABLE_NAME = 'langchain_pg_embedding'

# Initialize Gemini API
os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

model = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    temperature=0.2,
)

# ----------------------
# Conversation Phase Tracking
# ----------------------
class ConversationPhase:
    """Track conversation phases to control question flow."""
    GREETING = "greeting"
    INTAKE = "intake"           # Initial symptom gathering (3-5 key facts)
    CLARIFICATION = "clarification"  # Optional follow-up (1-2 questions max)
    REMEDY = "remedy"           # Provide recommendation
    FOLLOWUP = "followup"       # Post-remedy discussion

# ----------------------
# Data Structure Analysis Helper
# ----------------------
def _analyze_data_structure(cursor) -> Dict[str, Any]:
    """Analyze the actual data in the table to understand different data types and patterns."""
    analysis = {
        "total_records": 0,
        "data_types_found": [],
        "sample_metadata_structures": [],
        "common_fields": set()
    }
    
    try:
        cursor.execute(f"SELECT COUNT(*) as count FROM {TABLE_NAME}")
        analysis["total_records"] = cursor.fetchone()['count']
        
        cursor.execute(f"""
            SELECT document, cmetadata, collection_id
            FROM {TABLE_NAME}
            ORDER BY RANDOM()
            LIMIT 10
        """)
        
        samples = cursor.fetchall()
        
        for sample in samples:
            metadata = sample.get('cmetadata', {})
            
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
            
            if isinstance(metadata, dict):
                source_type = metadata.get('source_type', metadata.get('source', 'unknown'))
                if source_type not in analysis["data_types_found"]:
                    analysis["data_types_found"].append(source_type)
                
                for key in metadata.keys():
                    analysis["common_fields"].add(key)
                
                if len(analysis["sample_metadata_structures"]) < 3:
                    analysis["sample_metadata_structures"].append({
                        "source_type": source_type,
                        "fields": list(metadata.keys()),
                        "sample_data": {k: str(v)[:50] for k, v in list(metadata.items())[:5]}
                    })
        
        analysis["common_fields"] = list(analysis["common_fields"])
        
    except Exception as e:
        print(f"Warning: Could not analyze data structure: {e}")
    
    return analysis

# ----------------------
# AI-Powered SQL Query Builder
# ----------------------
def _build_sql_with_ai(user_question: str, data_analysis: Dict[str, Any], history: List[Dict[str, str]] = None) -> Tuple[str, List[Any], Dict[str, Any]]:
    """Use AI to understand the question and build an appropriate SQL query."""
    
    # Resolve pronouns from history if needed
    resolved_question = user_question
    if history:
        history_context = "\n".join([f"{h['role'].upper()}: {h['message']}" for h in history[-3:]])
        
        pronoun_pattern = r'\b(he|she|him|her|his|hers|it|its|they|them|their|that|this|also|too)\b'
        if re.search(pronoun_pattern, user_question.lower()):
            try:
                resolve_prompt = f"""Given this conversation history:
{history_context}

Current question: {user_question}

The question contains pronouns or reference words (he/she/it/also/too/etc).
Rewrite it to be self-contained by replacing references with the actual subject from history.
Return ONLY the rewritten question, nothing else.

Example:
History: "USER: who is H001"
Question: "what is his contact number"
Rewritten: "what is the contact number of H001"

Rewritten question:"""
                
                resolve_response = model.invoke([HumanMessage(content=resolve_prompt)])
                resolved_question = resolve_response.content.strip()
                print(f"🔄 Resolved: '{user_question}' -> '{resolved_question}'")
            except Exception as e:
                print(f"⚠️ Pronoun resolution failed: {e}")
    
    system_prompt = f"""You are an expert SQL query builder for PostgreSQL table '{TABLE_NAME}'.

TABLE STRUCTURE:
- id: UUID
- collection_id: UUID  
- document: TEXT (actual content - may contain MULTIPLE records separated by '--- RECORD BREAK ---')
- cmetadata: JSONB (metadata: filename, Patient ID, Record Number, etc.)

DATA ANALYSIS:
Total records: {data_analysis['total_records']}
Common fields: {', '.join(data_analysis['common_fields'])}

QUERY BUILDING RULES:
1. Output ONLY valid JSON: {{"sql": "...", "params": [...], "reasoning": "..."}}
2. Use %s placeholders for parameters
3. Search strategy:
   - For patient IDs (H001, H002): Search ONLY for the ID
     WHERE document ILIKE %s OR cmetadata::text ILIKE %s
     params: ["%H001%", "%H001%"]
   
   - For names: Search for each name part
     WHERE (document ILIKE %s OR cmetadata::text ILIKE %s) 
     AND (document ILIKE %s OR cmetadata::text ILIKE %s)
     params: ["%john%", "%john%", "%doe%", "%doe%"]

4. Return ALL documents that MIGHT contain the match (we filter later)
5. Order by: (cmetadata->>'created_at')::bigint DESC NULLS LAST
6. Include LIMIT (default 100)
7. ONLY SELECT queries

IMPORTANT: Search for terms explicitly in the question. Don't add extra search terms.

OUTPUT: Pure JSON only, no markdown."""

    human_prompt = f"""Question: {resolved_question}

Build an SQL query to find this information.

JSON:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        response = model.invoke(messages)
        response_text = response.content.strip()
        
        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, flags=re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_match = re.search(r'(\{.*\})', response_text, flags=re.DOTALL)
            json_text = json_match.group(1) if json_match else response_text
        
        parsed = json.loads(json_text)
        
        sql = parsed.get('sql', '')
        params = parsed.get('params', [])
        reasoning = parsed.get('reasoning', '')
        
        # Validate
        if not sql or not isinstance(sql, str):
            raise ValueError("Invalid SQL")
        
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            raise ValueError("Only SELECT allowed")
        
        if 'LIMIT' not in sql_upper:
            sql = sql.strip().rstrip(';') + ' LIMIT 100'
        
        debug = {
            'method': 'ai_built',
            'reasoning': reasoning,
            'resolved_question': resolved_question
        }
        
        return sql, params, debug
        
    except Exception as e:
        print(f"AI query building failed: {e}")
        return _simple_fallback_query(resolved_question)

# ----------------------
# Simple Fallback Query Builder
# ----------------------
def _simple_fallback_query(user_question: str) -> Tuple[str, List[Any], Dict[str, Any]]:
    """Fallback query builder when AI fails."""
    words = re.findall(r'\w+', user_question.lower())
    stop_words = {'what', 'where', 'when', 'who', 'how', 'is', 'are', 'the', 'a', 'an', 
                   'can', 'find', 'show', 'tell', 'give', 'me', 'please', 'about'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    if not keywords:
        sql = f"""
            SELECT id, document, cmetadata, collection_id 
            FROM {TABLE_NAME}
            ORDER BY (cmetadata->>'created_at')::bigint DESC NULLS LAST
            LIMIT 50
        """
        params = []
    else:
        search_conditions = []
        params = []
        
        for keyword in keywords[:3]:
            pattern = f"%{keyword}%"
            search_conditions.append(
                "(document ILIKE %s OR cmetadata::text ILIKE %s)"
            )
            params.extend([pattern, pattern])
        
        where_clause = " OR ".join(search_conditions)
        sql = f"""
            SELECT id, document, cmetadata, collection_id 
            FROM {TABLE_NAME}
            WHERE {where_clause}
            ORDER BY (cmetadata->>'created_at')::bigint DESC NULLS LAST
            LIMIT 100
        """
    
    debug = {
        'method': 'fallback',
        'keywords': keywords,
        'reason': 'AI builder failed or returned invalid query'
    }
    
    return sql, params, debug

# ----------------------
# Format Results for User
# ----------------------
def _format_results(rows: List[Dict], sql: str, params: List, debug: Dict, user_question: str) -> Tuple[str, List[str]]:
    """Format query results - return RAW data for LLM to process and list of filenames."""
    if not rows:
        return "NO_RESULTS_FOUND", []
    
    # Extract search terms from question
    search_terms = []
    
    # Extract patient IDs (H001, H002, etc.)
    patient_id_matches = re.findall(r'\b(H\d{3,4})\b', user_question, re.IGNORECASE)
    if patient_id_matches:
        search_terms.extend([pid.upper() for pid in patient_id_matches])
    
    # Extract names
    question_words = ['what', 'who', 'where', 'when', 'how', 'tell', 'about', 'give', 'show', 'find', 'details', 'information', 'is', 'are', 'the', 'a', 'an', 'me', 'for', 'of']
    words = user_question.lower().split()
    name_parts = [w.strip('?,.:;!').title() for w in words if w.lower() not in question_words and len(w) > 2 and not re.match(r'^h\d+$', w.lower())]
    if name_parts:
        search_terms.extend(name_parts)
    
    matching_records = []
    filenames_set = set()
    
    for row in rows:
        doc = row.get('document', '').strip()
        if not doc:
            continue
        
        # Extract filename from metadata
        metadata = row.get('cmetadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        filename = metadata.get('filename', '')
        if filename:
            filenames_set.add(filename)
        
        # Split by RECORD BREAK if it exists
        if '--- RECORD BREAK ---' in doc:
            records = doc.split('--- RECORD BREAK ---')
        else:
            records = [doc]
        
        for record in records:
            record = record.strip()
            if not record:
                continue
            
            # If search terms exist, check if this record matches
            if search_terms:
                record_upper = record.upper()
                
                patient_ids_in_record = [pid for pid in search_terms if re.match(r'^H\d+$', pid)]
                names_in_record = [name for name in search_terms if not re.match(r'^H\d+$', name)]
                
                matches = False
                
                if patient_ids_in_record:
                    for pid in patient_ids_in_record:
                        if f"Patient ID: {pid}" in record or f"patient_id: {pid.lower()}" in record.lower():
                            matches = True
                            break
                
                elif names_in_record:
                    if all(name.upper() in record_upper for name in names_in_record):
                        matches = True
                
                if not matches:
                    continue
            
            matching_records.append(record)
    
    if not matching_records:
        return "NO_RESULTS_FOUND", []
    
    return "\n\n=== RECORD ===\n\n".join(matching_records), list(filenames_set)

# ----------------------
# Main Tool: query_knowledge_base
# ----------------------
@tool
def query_knowledge_base(user_question: str) -> str:
    """
    Intelligent query tool that understands the question and retrieves relevant data
    from the knowledge base using AI-generated SQL queries.
    
    Args:
        user_question: Natural language question about the data
        
    Returns:
        Formatted results with relevant data and metadata
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        data_analysis = _analyze_data_structure(cursor)
        sql, params, debug = _build_sql_with_ai(user_question, data_analysis)
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as exec_error:
            print(f"Query execution failed: {exec_error}")
            sql, params, debug = _simple_fallback_query(user_question)
            debug['execution_error'] = str(exec_error)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        
        result, filenames = _format_results(rows, sql, params, debug, user_question)
        
        if filenames and result != "NO_RESULTS_FOUND":
            debug['filenames'] = filenames
        
        cursor.close()
        conn.close()
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error querying knowledge base: {str(e)}"

# ----------------------
# Conversation Phase Detection
# ----------------------
def _detect_conversation_phase(history: List[Dict[str, str]], user_input: str) -> str:
    """Detect what phase of conversation we're in."""
    if not history:
        # Check if it's a greeting
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(pattern in user_input.lower() for pattern in greeting_patterns):
            return ConversationPhase.GREETING
        return ConversationPhase.INTAKE
    
    # Count actual symptom-gathering questions (questions about symptoms, not rhetorical)
    symptom_question_count = 0
    for msg in history:
        if msg['role'] == 'ai':
            text = msg['message'].lower()
            # Only count questions that are actually gathering symptom information
            symptom_indicators = [
                'where', 'when', 'what time', 'how long', 'describe',
                'sensation', 'feeling', 'experience', 'intensity',
                'better', 'worse', 'modalities', 'aggravation', 'amelioration'
            ]
            if '?' in text and any(indicator in text for indicator in symptom_indicators):
                symptom_question_count += 1
    
    # Check if we have enough information gathered
    has_location = any('where' in msg['message'].lower() for msg in history if msg['role'] == 'ai')
    has_sensation = any('sensation' in msg['message'].lower() or 'feel' in msg['message'].lower() for msg in history if msg['role'] == 'ai')
    has_modality = any('better' in msg['message'].lower() or 'worse' in msg['message'].lower() for msg in history if msg['role'] == 'ai')
    
    gathered_facts = sum([has_location, has_sensation, has_modality])
    
    # Phase determination
    if symptom_question_count >= 5 or gathered_facts >= 3:
        return ConversationPhase.REMEDY
    elif symptom_question_count >= 3:
        return ConversationPhase.CLARIFICATION
    else:
        return ConversationPhase.INTAKE

# ----------------------
# Generate Symptom Options
# ----------------------
def _generate_symptom_options(context: str = "") -> str:
    """Generate numbered symptom options for user convenience."""
    common_symptoms = [
        "Fever",
        "Cold/Cough",
        "Headache",
        "Stomach pain",
        "Body ache",
        "Fatigue",
        "Anxiety",
        "Skin issues",
        "Sleep problems",
        "Other (describe)"
    ]
    
    options_text = "\n\n**Common symptoms (you can type the number or describe in your own words):**\n"
    for i, symptom in enumerate(common_symptoms, 1):
        options_text += f"{i}. {symptom}\n"
    
    return options_text

# ----------------------
# Parse User Symptom Choice
# ----------------------
def _parse_symptom_choice(user_input: str) -> str:
    """Parse user input if they selected a numbered option."""
    symptom_map = {
        "1": "fever",
        "2": "cold or cough",
        "3": "headache",
        "4": "stomach pain",
        "5": "body ache",
        "6": "fatigue",
        "7": "anxiety",
        "8": "skin issues",
        "9": "sleep problems",
        "10": "other symptoms"
    }
    
    # Check if user just typed a number
    stripped = user_input.strip()
    if stripped in symptom_map:
        return symptom_map[stripped]
    
    return user_input  # Return original if not a number

# ----------------------
# Agent Setup
# ----------------------
tools = [query_knowledge_base]
model_with_tools = model.bind_tools(tools)

def run_agent(user_input: str, history: List[Dict[str, str]] = None, max_iterations: int = 5, user_id: Optional[int] = None, vector_store = None) -> Tuple[str, List[str]]:
    """Run the agent with conversation history context. Returns (response, filenames)."""
    
    history = history or []
    history_context = ""
    if history:
        history_context = "\n".join([f"{h['role'].upper()}: {h['message']}" for h in history[-5:]])
    
    # Parse symptom choice if user selected a number
    parsed_input = _parse_symptom_choice(user_input)
    if parsed_input != user_input:
        print(f"📋 User selected option: {user_input} -> {parsed_input}")
        user_input = parsed_input
    
    # Detect conversation phase
    current_phase = _detect_conversation_phase(history, user_input)
    print(f"🔄 Conversation phase: {current_phase}")
    
    # Get personalization
    username = None
    nickname = None
    custom_instructions = None
    
    if user_id and vector_store:
        try:
            if hasattr(vector_store, 'SessionLocal') and hasattr(vector_store, 'UserORM'):
                with vector_store.SessionLocal() as db:
                    user = db.query(vector_store.UserORM).filter_by(id=user_id).first()
                    if user:
                        username = user.username
                    
                    admin_user = db.query(vector_store.UserORM).filter_by(role='admin').order_by(vector_store.UserORM.id.asc()).first()
                    if admin_user and hasattr(vector_store, 'get_user_personalization'):
                        personalization = vector_store.get_user_personalization(admin_user.id)
                        if personalization:
                            if personalization.get('nickname'):
                                nickname = personalization['nickname'].strip()
                            if personalization.get('custom_instructions'):
                                custom_instructions = personalization['custom_instructions'].strip()
        except Exception as e:
            print(f"⚠️ Could not retrieve user info for user_id {user_id}: {e}")
    
    # Track filenames from tool results
    agent_filenames = []
    
    # Build system prompt based on phase
    system_content = ""
    
    # Add identity first
    if nickname:
        system_content += f"""## Your Identity:
You are {nickname}. Always introduce yourself as {nickname}.

"""
    
    # Add custom instructions if available
    if custom_instructions:
        system_content += f"""## Custom Instructions:
{custom_instructions}

"""
    
    # Phase-specific instructions
    if current_phase == ConversationPhase.GREETING:
        system_content += """## Current Phase: GREETING
Greet the user warmly and ask what brings them here today.
Optionally, you can provide numbered symptom options to make it easier for them.
"""
        # Add symptom options for first interaction
        system_content += _generate_symptom_options()
    
    elif current_phase == ConversationPhase.INTAKE:
        system_content += """## Current Phase: INTAKE (Initial Information Gathering)
Your goal: Gather 3-5 key facts about the user's condition.

MUST GATHER:
1. Primary symptom/complaint
2. Location of discomfort
3. Sensation quality (burning, aching, sharp, etc.)
4. When it started / Duration
5. What makes it better or worse

RULES:
- Ask ONE focused question at a time
- Be conversational and empathetic
- Maximum 5 questions in this phase
- After gathering these facts, move to remedy phase
"""

    elif current_phase == ConversationPhase.CLARIFICATION:
        system_content += """## Current Phase: CLARIFICATION
You have basic information. Ask 1-2 clarifying questions if needed.

RULES:
- MAXIMUM 2 questions in this phase
- Only ask if truly necessary for remedy selection
- Then MUST provide remedy

Example clarifying questions:
- "Is the pain worse at night or during the day?"
- "Does warmth help or make it worse?"
"""

    elif current_phase == ConversationPhase.REMEDY:
        system_content += """## Current Phase: REMEDY RECOMMENDATION
You have gathered enough information. NOW PROVIDE A REMEDY.

REQUIREMENTS:
- State the recommended homeopathic remedy
- Explain why it matches their symptoms
- Provide dosage instructions (e.g., "30C potency, 3 pellets under tongue, 3 times daily")
- Mention when to expect improvement
- DO NOT ASK MORE QUESTIONS (unless user asks follow-up)

REMEDY FORMAT:
"Based on your symptoms, I recommend [Remedy Name] [Potency].

This remedy is suitable because [reason matching symptoms].

Dosage: [specific instructions]

You should notice improvement within [timeframe]."
"""
    
    # Add username info
    if username:
        system_content += f"""
## User Information:
User's username: {username}
"""
    
    # Tool usage
    system_content += """
## Tool Usage:
- Use query_knowledge_base to search patient records when asked about specific patients
- After tool results, provide clear summary
- If tool returns "NO_RESULTS_FOUND", inform user clearly

## Response Requirements:
- Be natural and conversational
- Show empathy
- Be concise but complete
- ALWAYS provide text output (empty responses not acceptable)
"""
    
    if history_context:
        system_content += f"\n## Recent Conversation:\n{history_context}\n"
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input)
    ]
    
    data_analysis = None
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*60}\n")
        
        response = model_with_tools.invoke(messages)
        print(f'📨 Model response type: {type(response)}')
        print(f'📨 Has tool_calls: {hasattr(response, "tool_calls") and bool(response.tool_calls)}')
        print(f'📨 Response content length: {len(response.content) if response.content else 0}')
        
        messages.append(response)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Model requested {len(response.tool_calls)} tool call(s)")
            
            for call in response.tool_calls:
                if call['name'] == 'query_knowledge_base':
                    print(f"\n🔍 Searching knowledge base with: {call['args'].get('user_question')}\n")
                    
                    try:
                        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
                        cursor = conn.cursor()
                        if not data_analysis:
                            data_analysis = _analyze_data_structure(cursor)
                        
                        sql, params, debug = _build_sql_with_ai(
                            call['args'].get('user_question'), 
                            data_analysis,
                            history=history
                        )
                        
                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                        
                        tool_result, filenames = _format_results(rows, sql, params, debug, call['args'].get('user_question'))
                        if filenames:
                            agent_filenames.extend(filenames)
                        
                        cursor.close()
                        conn.close()
                    except Exception as e:
                        print(f"❌ Database query error: {e}")
                        tool_result = "NO_RESULTS_FOUND"
                        filenames = []
                    
                    print(f"\n📦 Tool Result Preview:\n{tool_result[:300]}...\n")
                    
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=call['id'])
                    )
                    print("🔄 Tool result added to conversation. Requesting final answer from LLM...\n")
            
            continue
        
        else:
            # No tool calls - this is the final answer
            print('\n✅ Final answer received from model')
            
            answer_text = response.content
            if isinstance(answer_text, list) and len(answer_text) > 0:
                if isinstance(answer_text[0], dict) and 'text' in answer_text[0]:
                    answer_text = answer_text[0]['text']
                else:
                    answer_text = str(answer_text)
            elif not isinstance(answer_text, str):
                answer_text = str(answer_text)
            
            if not answer_text or answer_text.strip() == "":
                print("⚠️ WARNING: Empty response - using fallback")
                
                # Try to extract from tool results
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage) and msg.content != "NO_RESULTS_FOUND":
                        lines = msg.content.split('\n')
                        patient_info = []
                        for line in lines[:15]:
                            if any(key in line for key in ['Patient ID:', 'Age:', 'Gender:', 'Chief Complaints:', 'Date of Visit:']):
                                patient_info.append(line.strip())
                        
                        if patient_info:
                            return "Patient details: " + " | ".join(patient_info), list(set(agent_filenames))
                        else:
                            return "Based on the records: " + ' '.join(lines[:5]), list(set(agent_filenames))
                
                return "I found information but couldn't format it properly. Please try again.", list(set(agent_filenames))
            
            print(f"✅ Returning final answer ({len(answer_text)} chars)")
            return answer_text, list(set(agent_filenames))
    
    return "I apologize, but I couldn't complete the request. Please try again.", list(set(agent_filenames))

# ----------------------
# Main Execution
# ----------------------
if __name__ == '__main__':
    print("🤖 AI-Powered Knowledge Base Query Tool")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
    else:
        question = input('\n💬 Enter your question: ')
    
    print(f"\n🎯 Processing: {question}\n")
    print("=" * 60)
    
    answer, filenames = run_agent(question)
    
    print("\n" + "=" * 60)
    print("✨ FINAL ANSWER:")
    print("=" * 60)
    print(answer)
    if filenames:
        print("\n📄 Sources:", ", ".join(filenames))