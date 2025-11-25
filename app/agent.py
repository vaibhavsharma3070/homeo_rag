# ai_sql_query_tool.py
# AI-driven SQL query builder for diverse data types in langchain_pg_embedding table
# The AI analyzes the question, understands the data structure, and builds optimized queries

import os
import sys
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Tuple

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
    temperature=0.0,
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
        
        # Check if question has pronouns
        pronoun_pattern = r'\b(he|she|him|her|his|hers|it|its|they|them|their|that|this)\b'
        if re.search(pronoun_pattern, user_question.lower()):
            try:
                resolve_prompt = f"""Given this conversation history:
{history_context}

Current question: {user_question}

Rewrite the question to be self-contained by replacing pronouns with the actual subject from history.
Return ONLY the rewritten question, nothing else.

Rewritten question:"""
                
                resolve_response = model.invoke([HumanMessage(content=resolve_prompt)])
                resolved_question = resolve_response.content.strip()
                logger.info(f"Resolved question: '{user_question}' -> '{resolved_question}'")
            except Exception as e:
                logger.warning(f"Pronoun resolution failed: {e}")
    
    system_prompt = f"""You are an expert SQL query builder for PostgreSQL table '{TABLE_NAME}'.

    TABLE STRUCTURE:
    - id: UUID
    - collection_id: UUID  
    - document: TEXT (actual content - may contain MULTIPLE records separated by '--- RECORD BREAK ---')
    - cmetadata: JSONB (metadata: filename, Patient ID, Record Number, etc.)

    CRITICAL: Each document may contain MULTIPLE patient records separated by '--- RECORD BREAK ---'

    DATA ANALYSIS:
    Total records: {data_analysis['total_records']}
    Common fields: {', '.join(data_analysis['common_fields'])}

    QUERY BUILDING RULES:
    1. Output ONLY valid JSON: {{"sql": "...", "params": [...], "reasoning": "..."}}
    2. Use %s placeholders for parameters
    3. NEVER assume data that isn't explicitly mentioned in the question
    4. Search strategy:
       - For patient IDs (H001, H002): Search ONLY for the ID
         WHERE document ILIKE %s OR cmetadata::text ILIKE %s
         params: ["%H001%", "%H001%"]
       
       - For "name of H001" type questions: Search ONLY for H001, NOT for names
         WHERE document ILIKE %s OR cmetadata::text ILIKE %s
         params: ["%H001%", "%H001%"]
       
       - For explicit names (e.g., "find John Doe"): 
         WHERE (document ILIKE %s OR cmetadata::text ILIKE %s) 
         AND (document ILIKE %s OR cmetadata::text ILIKE %s)
         params: ["%john%", "%john%", "%doe%", "%doe%"]
    
    5. Return ALL documents that MIGHT contain the match (we filter later)
    6. Order by: (cmetadata->>'created_at')::bigint DESC NULLS LAST
    7. Include LIMIT (default 100)
    8. ONLY SELECT queries - no INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE

    IMPORTANT: Don't add search terms that aren't in the question. If asked about "H001", search ONLY for "H001", not for names.

    OUTPUT: Pure JSON only, no markdown."""

    human_prompt = f"""Question: {resolved_question}

Build an SQL query. Remember: ONLY search for terms explicitly mentioned in the question.

JSON:"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        response = model.invoke(messages)
        print('response =========================================== ',response)
        response_text = response.content.strip()
        print('response_text =========================================== ',response_text)
        
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
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'TRUNCATE', 'ALTER', 'CREATE']
        for keyword in forbidden:
            if keyword in sql_upper:
                raise ValueError(f"Forbidden keyword detected: {keyword}")
        
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
    """Format query results - extract and return ONLY the relevant records."""
    if not rows:
        return "I couldn't find any information about that in the knowledge base."
    
    # Extract search terms from question to filter results
    search_terms = []
    
    # Extract patient IDs (H001, H002, etc.)
    patient_id_matches = re.findall(r'\b(H\d{3,4})\b', user_question, re.IGNORECASE)
    if patient_id_matches:
        search_terms.extend([pid.upper() for pid in patient_id_matches])
    
    # Extract names (John Doe, etc.)
    # Remove common question words first
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
            # Single record
            records = [doc]
        
        # Check each record
        for record in records:
            record = record.strip()
            if not record:
                continue
            
            # If search terms exist, check if this record matches
            if search_terms:
                record_upper = record.upper()
                # Check if ALL search terms are in this specific record (for names)
                # OR if ANY patient ID matches (for IDs)
                
                patient_ids_in_record = [pid for pid in search_terms if re.match(r'^H\d+$', pid)]
                names_in_record = [name for name in search_terms if not re.match(r'^H\d+$', name)]
                
                matches = False
                
                # If patient ID searched, match exactly
                if patient_ids_in_record:
                    for pid in patient_ids_in_record:
                        if f"Patient ID: {pid}" in record or f"patient_id: {pid.lower()}" in record.lower():
                            matches = True
                            break
                
                # If name searched, all parts must be present
                elif names_in_record:
                    if all(name.upper() in record_upper for name in names_in_record):
                        matches = True
                
                if not matches:
                    continue
            
            # Extract key information from the record
            lines = record.split('\n')
            record_data = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    record_data[key] = value
            
            # Build a clean summary
            summary_parts = []
            
            # Add key fields
            priority_fields = ['Record Number', 'Patient ID', 'Name', 'Age', 'Gender', 'Date of Visit', 
                             'Chief Complaints', 'History of Present Illness', 'Remedy Prescribed', 
                             'Potency', 'Dosage', 'Follow-up Status']
            
            for field in priority_fields:
                if field in record_data and record_data[field]:
                    summary_parts.append(f"{field}: {record_data[field]}")
            
            # Add other fields (limit to avoid clutter)
            other_fields = [k for k in record_data.keys() if k not in priority_fields]
            for field in other_fields[:5]:  # Limit to 5 additional fields
                if record_data[field]:
                    summary_parts.append(f"{field}: {record_data[field]}")
            
            if summary_parts:
                matching_records.append('\n'.join(summary_parts))
    
    # Return results
    if not matching_records:
        return "I couldn't find any information about that in the knowledge base."
    
    total = len(matching_records)
    
    if total == 1:
        return f"Here's what I found:\n\n{matching_records[0]}"
    else:
        # Multiple matches - show them numbered
        result_parts = [f"I found {total} matching record(s):\n"]
        for idx, record in enumerate(matching_records[:5], 1):  # Show max 5
            result_parts.append(f"\n{idx}. {record}")
        
        if total > 5:
            result_parts.append(f"\n\n... and {total - 5} more records.")
        
        return '\n'.join(result_parts)

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

def run_agent(user_input: str, history: List[Dict[str, str]] = None, max_iterations: int = 5) -> str:
    """Run the agent with conversation history context."""
    
    history_context = ""
    if history:
        history_context = "\n".join([f"{h['role'].upper()}: {h['message']}" for h in history[-3:]])
    
    system_content = """You are a helpful medical records assistant with access to a patient database.

    CRITICAL RULES:
    1. ALWAYS use the query_knowledge_base tool to search
    2. The tool returns FILTERED results - only the specific records requested
    3. Give direct, natural answers
    4. Do NOT mention: chunks, files, documents, sources, databases, metadata, records, or technical terms
    5. If user asks about a patient ID (like H010), only talk about THAT patient
    6. If multiple records match, present them clearly
    7. Check conversation history for pronoun references (he/she/it)

    When the tool returns "I found X matching record(s)", trust that it filtered correctly.
    Present the information naturally without technical details."""

    if history_context:
        system_content += f"\n\nRecent conversation:\n{history_context}"
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_input)
    ]

    # Pass history to data analysis
    data_analysis = None
    
    for iteration in range(max_iterations):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            for call in response.tool_calls:
                if call['name'] == 'query_knowledge_base':
                    print(f"\n🔍 Searching knowledge base with: {call['args'].get('user_question')}\n")
                    
                    # Get data analysis with history context
                    try:
                        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
                        cursor = conn.cursor()
                        if not data_analysis:
                            data_analysis = _analyze_data_structure(cursor)
                        
                        # Build query with history
                        sql, params, debug = _build_sql_with_ai(
                            call['args'].get('user_question'), 
                            data_analysis,
                            history=history
                        )
                        
                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                        
                        # Format results with filtering
                        tool_result = _format_results(rows, sql, params, debug, call['args'].get('user_question'))
                        
                        cursor.close()
                        conn.close()
                    except Exception as e:
                        logger.error(f"Database query error: {e}")
                        tool_result = f"I encountered an error while searching: {str(e)}"
                    
                    print(f"\n📦 Tool Result Preview:\n{tool_result[:300]}...\n")
                    
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=call['id'])
                    )
        else:
            # Final answer
            return response.content

    return "I couldn't find a complete answer to your question."

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