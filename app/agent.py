# ai_sql_query_tool.py
# AI-driven SQL query builder for diverse data types in langchain_pg_embedding table
# The AI analyzes the question, understands the data structure, and builds optimized queries

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
# Data Structure Analysis Helper
# ----------------------
def _analyze_data_structure(cursor) -> Dict[str, Any]:
    """
    Analyze the actual data in the table to understand different data types and patterns.
    Returns a summary of data patterns for the AI to understand.
    """
    analysis = {
        "total_records": 0,
        "data_types_found": [],
        "sample_metadata_structures": [],
        "common_fields": set()
    }
    
    try:
        # Get total count
        cursor.execute(f"SELECT COUNT(*) as count FROM {TABLE_NAME}")
        analysis["total_records"] = cursor.fetchone()['count']
        
        # Sample diverse records to understand data patterns
        cursor.execute(f"""
            SELECT document, cmetadata, collection_id
            FROM {TABLE_NAME}
            ORDER BY RANDOM()
            LIMIT 10
        """)
        
        samples = cursor.fetchall()
        
        for sample in samples:
            metadata = sample.get('cmetadata', {})
            
            # Parse metadata if it's a string
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
            
            if isinstance(metadata, dict):
                # Identify data type
                source_type = metadata.get('source_type', metadata.get('source', 'unknown'))
                if source_type not in analysis["data_types_found"]:
                    analysis["data_types_found"].append(source_type)
                
                # Collect common fields
                for key in metadata.keys():
                    analysis["common_fields"].add(key)
                
                # Store sample structure
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
        
        # Safety checks
        # forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER', 'CREATE']
        # for keyword in forbidden:
        #     if keyword in sql_upper:
        #         raise ValueError(f"Forbidden keyword detected: {keyword}")
        
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
    """
    Fallback query builder when AI fails.
    Creates a basic search across document and metadata.
    """
    # Extract keywords
    words = re.findall(r'\w+', user_question.lower())
    stop_words = {'what', 'where', 'when', 'who', 'how', 'is', 'are', 'the', 'a', 'an', 
                   'can', 'find', 'show', 'tell', 'give', 'me', 'please', 'about'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    if not keywords:
        # Return recent records
        sql = f"""
            SELECT id, document, cmetadata, collection_id 
            FROM {TABLE_NAME}
            ORDER BY (cmetadata->>'created_at')::bigint DESC NULLS LAST
            LIMIT 50
        """
        params = []
    else:
        # Search in both document and metadata
        search_conditions = []
        params = []
        
        for keyword in keywords[:3]:  # Limit to 3 keywords
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
def _format_results(rows: List[Dict], sql: str, params: List, debug: Dict, user_question: str) -> str:
    """Format query results - return RAW data for LLM to process."""
    if not rows:
        return "NO_RESULTS_FOUND"
    
    # Extract search terms from question
    search_terms = []
    
    # Extract patient IDs (H001, H002, etc.)
    patient_id_matches = re.findall(r'\b(H\d{3,4})\b', user_question, re.IGNORECASE)
    if patient_id_matches:
        search_terms.extend([pid.upper() for pid in patient_id_matches])
    
    # Extract names (John Doe, etc.)
    question_words = ['what', 'who', 'where', 'when', 'how', 'tell', 'about', 'give', 'show', 'find', 'details', 'information', 'is', 'are', 'the', 'a', 'an', 'me', 'for', 'of']
    words = user_question.lower().split()
    name_parts = [w.strip('?,.:;!').title() for w in words if w.lower() not in question_words and len(w) > 2 and not re.match(r'^h\d+$', w.lower())]
    if name_parts:
        search_terms.extend(name_parts)
    
    # Collect all matching records
    matching_records = []
    
    for row in rows:
        doc = row.get('document', '').strip()
        if not doc:
            continue
        
        # Split by RECORD BREAK if it exists
        if '--- RECORD BREAK ---' in doc:
            records = doc.split('--- RECORD BREAK ---')
        else:
            records = [doc]
        
        # Check each record
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
            
            # Add the raw record
            matching_records.append(record)
    
    # Return RAW data for LLM to process
    if not matching_records:
        return "NO_RESULTS_FOUND"
    
    # Return records separated by delimiter
    return "\n\n=== RECORD ===\n\n".join(matching_records)

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
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        # Analyze data structure
        data_analysis = _analyze_data_structure(cursor)
        
        # Build SQL query using AI
        sql, params, debug = _build_sql_with_ai(user_question, data_analysis)
        
        # Execute query
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as exec_error:
            print(f"Query execution failed: {exec_error}")
            # Try fallback query
            sql, params, debug = _simple_fallback_query(user_question)
            debug['execution_error'] = str(exec_error)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        
        # Format and return results
        result = _format_results(rows, sql, params, debug, user_question)
        
        cursor.close()
        conn.close()
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error querying knowledge base: {str(e)}"

# ----------------------
# Agent Setup
# ----------------------
tools = [query_knowledge_base]
model_with_tools = model.bind_tools(tools)

def run_agent(user_input: str, history: List[Dict[str, str]] = None, max_iterations: int = 5, user_id: Optional[int] = None, vector_store = None) -> str:
    """Run the agent with conversation history context."""
    
    history_context = ""
    if history:
        history_context = "\n".join([f"{h['role'].upper()}: {h['message']}" for h in history[-3:]])
    
    system_content = """You are a helpful medical records assistant with access to a patient database.

    **CRITICAL REQUIREMENT:**
    After you receive tool results from query_knowledge_base, you MUST ALWAYS write a natural language summary of the data.
    NEVER return empty content. Even if the data is minimal, you must describe what you found.

    STRICT RULES:
    1. ALWAYS use the query_knowledge_base tool first to search for information
    2. After receiving tool results, you MUST write at least 2-3 sentences describing the data
    3. If the tool returns patient records, extract key details (name, age, condition, treatment)
    4. Format your response naturally, as if explaining to a colleague
    5. If tool returns "NO_RESULTS_FOUND", state clearly: "I don't have information about that patient in the database"
    6. Please don't say Good morning each time

    REQUIRED OUTPUT FORMAT:
    ✅ Good: "Patient H002 is a 41-year-old male who visited on August 12, 2024 with multiple filiform warts..."
    ✅ Good: "The patient presented with warts for 8 months causing cosmetic discomfort..."
    ❌ Bad: Empty response
    ❌ Bad: Just metadata without description

    **YOU MUST PROVIDE TEXT OUTPUT. EMPTY RESPONSES ARE NOT ACCEPTABLE.**"""

    if history_context:
        system_content += f"\n\nRecent conversation:\n{history_context}\n\nUse this context to understand pronouns and references."
    
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
                        print('sql ===========================================', sql)
                        print('params ===========================================', params)

                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                        
                        tool_result = _format_results(rows, sql, params, debug, call['args'].get('user_question'))
                        print('tool_result ===========================================', tool_result[:500])

                        cursor.close()
                        conn.close()
                    except Exception as e:
                        print(f"❌ Database query error: {e}")
                        tool_result = "NO_RESULTS_FOUND"
                    
                    print(f"\n📦 Tool Result Preview:\n{tool_result[:300]}...\n")
                    
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=call['id'])
                    )
                    print("🔄 Tool result added to conversation. Requesting final answer from LLM...\n")
            
            continue
                    
        else:
            # No tool calls - this is the final answer
            print('\n✅ Final answer received from model')
            print('response.content ===========================================', response.content)
            print(f'📝 Response content: "{response.content[:200]}..."')
            
            if not response.content or response.content.strip() == "":
                print("⚠️ WARNING: LLM returned empty content - forcing retry with explicit instructions")
                
                # Find the most recent tool result
                tool_result_content = None
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage) and msg.content != "NO_RESULTS_FOUND":
                        tool_result_content = msg.content
                        break
                
                if tool_result_content:
                    # Force a response with very explicit prompt
                    retry_message = HumanMessage(content=f"""You returned an empty response. This is not acceptable.

        Here is the patient data you received from the tool:
        {tool_result_content[:500]}

        YOU MUST write a 2-3 sentence summary of this patient information RIGHT NOW.
        Include: Patient ID, age, main complaint, and any treatment mentioned.

        Write the summary now:""")
                    
                    messages.append(retry_message)
                    
                    try:
                        print("🔄 Sending retry request with explicit instructions...")
                        retry_response = model_with_tools.invoke(messages)
                        
                        print(f"📨 Retry response content: '{retry_response.content}'")
                        
                        if retry_response.content and retry_response.content.strip():
                            print(f"✅ Retry successful! Got {len(retry_response.content)} chars")
                            return retry_response.content.strip()
                        else:
                            print("❌ Retry also returned empty - using fallback")
                    except Exception as retry_error:
                        print(f"❌ Retry failed with error: {retry_error}")
                
                # Final fallback - extract from tool results
                print("⚠️ Using manual fallback to extract patient info")
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage) and msg.content != "NO_RESULTS_FOUND":
                        # Parse the tool result and create a basic summary
                        lines = msg.content.split('\n')
                        patient_info = []
                        for line in lines[:15]:  # First 15 lines usually have key info
                            if any(key in line for key in ['Patient ID:', 'Age:', 'Gender:', 'Chief Complaints:', 'Date of Visit:']):
                                patient_info.append(line.strip())
                        
                        if patient_info:
                            return "Patient details: " + " | ".join(patient_info)
                        else:
                            return "Based on the records: " + ' '.join(lines[:5])
                
                return "I found patient records but couldn't generate a proper summary. Please try again."
            
            print(f"✅ Returning final answer ({len(response.content)} chars)")
            return response.content

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
    
    answer = run_agent(question)
    
    print("\n" + "=" * 60)
    print("✨ FINAL ANSWER:")
    print("=" * 60)
    print(answer)