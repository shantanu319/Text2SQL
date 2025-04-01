import os
import json
import sqlite3
import argparse
import re
from datasets import Dataset
from tqdm import tqdm  # Add progress bar
from collections import defaultdict

# System prompt for SQL generation
SYSTEM_PROMPT = """You are a helpful assistant that converts natural language questions into SQL queries.
Given a database schema and a question, your task is to generate a valid SQL query that answers the question.
Analyze the schema carefully and write a precise SQL query without any explanations or commentary.
Use simple direct queries where possible and avoid unnecessary table aliases or joins."""

# Format for user messages
USER_MESSAGE_FORMAT = """{schema}

Please answer the following question using the tables above.
{question}"""

# Qwen chat format
QWEN_CHAT_FORMAT = """<im_start>user
{message}<im_end>
<im_start>assistant
{response}<im_end>"""

def simplify_sql_query(sql, db_id=None):
    """
    Simplify SQL queries by removing unnecessary table aliases and joins when possible.
    This creates a cleaner version of complex queries to help the model learn both styles.
    """
    if not sql:
        return sql
    
    # Convert to lowercase for consistent processing
    sql_lower = sql.lower()
    
    # Make a copy of the original SQL
    simplified_sql = sql
    
    # 1. Check if this is a simple query with just one table but using aliases
    if " as t1" in sql_lower and " join " not in sql_lower:
        # Remove the table alias
        simplified_sql = re.sub(r'\s+as\s+t\d', '', simplified_sql, flags=re.IGNORECASE)
        # Replace any column references with aliases
        simplified_sql = re.sub(r't\d\.([a-zA-Z0-9_]+)', r'\1', simplified_sql, flags=re.IGNORECASE)
    
    # 2. Handle simple self-joins that can be eliminated
    join_match = re.search(r'from\s+([a-z0-9_]+)\s+as\s+t1\s+join\s+\1\s+as\s+t2\s+on\s+t1\.([a-z0-9_]+)\s*=\s*t2\.([a-z0-9_]+)', sql_lower)
    if join_match and not re.search(r'where.*t1\..*t2\.', sql_lower) and not re.search(r'having.*t1\..*t2\.', sql_lower):
        # This appears to be a self-join that might be unnecessary
        table_name = join_match.group(1)
        simplified_sql = re.sub(
            r'from\s+([a-z0-9_]+)\s+as\s+t1\s+join\s+\1\s+as\s+t2\s+on\s+t1\.[a-z0-9_]+\s*=\s*t2\.[a-z0-9_]+',
            f'from {table_name}',
            simplified_sql,
            flags=re.IGNORECASE
        )
        # Replace column references
        simplified_sql = re.sub(r't1\.([a-zA-Z0-9_]+)', r'\1', simplified_sql, flags=re.IGNORECASE)
        simplified_sql = re.sub(r't2\.([a-zA-Z0-9_]+)', r'\1', simplified_sql, flags=re.IGNORECASE)
    
    # 3. Standardize formatting for better learning
    simplified_sql = re.sub(r'\s+', ' ', simplified_sql).strip()
    
    # Only return the simplified version if it's different and still valid SQL
    if simplified_sql != sql and simplified_sql.lower().startswith(('select', 'with')):
        return simplified_sql
    
    return sql

def normalize_table_aliases(sql):
    """
    Normalize table aliases to be consistent with best practices:
    - Use meaningful table aliases instead of T1, T2 when possible
    - Use lowercase for improved readability
    """
    if not sql:
        return sql
    
    # Don't modify simple queries without JOINs or aliases
    if " as " not in sql.lower() or " join " not in sql.lower():
        return sql
    
    normalized_sql = sql
    
    # Extract table names and their aliases
    table_alias_pattern = r'from\s+([a-z0-9_]+)\s+as\s+([a-z0-9_]+)|join\s+([a-z0-9_]+)\s+as\s+([a-z0-9_]+)'
    
    # Find all table-alias pairs
    matches = re.finditer(table_alias_pattern, sql.lower(), re.IGNORECASE)
    table_aliases = {}
    
    for match in matches:
        if match.group(1) and match.group(2):  # FROM clause
            table = match.group(1)
            alias = match.group(2)
            table_aliases[alias] = table
        elif match.group(3) and match.group(4):  # JOIN clause
            table = match.group(3)
            alias = match.group(4)
            table_aliases[alias] = table
    
    # If we have t1, t2, etc. style aliases, convert them to more meaningful ones
    t_style_aliases = [alias for alias in table_aliases if re.match(r't\d+', alias, re.IGNORECASE)]
    
    if t_style_aliases:
        # Only apply this transformation to queries that use t1, t2 style
        
        # Create new replacement for each t-style alias
        new_aliases = {}
        for alias in t_style_aliases:
            table = table_aliases[alias]
            # Use first character of table name as an alias prefix
            prefix = table[0].lower()
            new_aliases[alias] = prefix
        
        # Replace t-style aliases with new aliases in the query
        for old_alias, new_alias in new_aliases.items():
            # Replace the AS clause
            pattern = r'(from|join)\s+([a-z0-9_]+)\s+as\s+' + old_alias + r'\b'
            replacement = r'\1 \2 AS ' + new_alias
            normalized_sql = re.sub(pattern, replacement, normalized_sql, flags=re.IGNORECASE)
            
            # Replace column references
            col_pattern = r'\b' + old_alias + r'\.([a-z0-9_]+)'
            col_replacement = new_alias + r'.\1'
            normalized_sql = re.sub(col_pattern, col_replacement, normalized_sql, flags=re.IGNORECASE)
    
    return normalized_sql

def get_schema_and_samples(db_path, max_rows=3):
    """Extract schema (CREATE TABLE statements) and sample rows from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table creation SQL statements from sqlite_master (skip internal tables)
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema_lines = []
    for table_name, create_sql in tables:
        if not create_sql or table_name.startswith("sqlite_"):
            continue  # skip internal or empty entries
        # Ensure the CREATE TABLE statement ends with a semicolon
        create_stmt = create_sql.strip()
        if not create_stmt.endswith(";"):
            create_stmt += ";"
        schema_lines.append(create_stmt)
        # Fetch sample rows
        try:
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {max_rows}")
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            rows = []
        # Append sample rows as an INSERT or comment
        if rows:
            # Prepare an INSERT statement with fetched rows
            col_count = len(rows[0])
            # We'll not list column names in INSERT for brevity; assume values align with table definition order
            insert_prefix = f"INSERT INTO {table_name} VALUES "
            values_list = []
            for row in rows:
                # Format each value for SQL (add quotes around text and escape single quotes)
                formatted_vals = []
                for val in row:
                    if isinstance(val, str):
                        # Escape single quotes by doubling them for SQL
                        safe_val = val.replace("'", "''")
                        formatted_vals.append(f"'{safe_val}'")
                    elif val is None:
                        formatted_vals.append("NULL")
                    else:
                        formatted_vals.append(str(val))
                values_list.append("(" + ", ".join(formatted_vals) + ")")
            insert_stmt = insert_prefix + ",\n       ".join(values_list) + ";"
            schema_lines.append(insert_stmt)
        else:
            # No rows available
            schema_lines.append("-- no sample rows for this table")
        schema_lines.append("")  # blank line to separate tables
    conn.close()
    # Join all schema lines into one schema text block
    schema_text = "\n".join(schema_lines).strip()
    return schema_text

def process_spider_for_finetuning(train_json_path, db_dir, output_path, balance_aliases=True):
    """Process Spider dataset into a format suitable for fine-tuning with chat templates."""
    # Load Spider training examples
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
        
    # Get list of available databases
    available_db_dirs = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    print(f"Found {len(available_db_dirs)} available database directories")
    
    # Prepare data for dataset creation
    processed_examples = []
    skipped_count = 0
    
    # Track alias usage statistics
    alias_stats = {"with_aliases": 0, "without_aliases": 0}
    
    # Add progress bar
    for example in tqdm(train_data, desc="Processing examples"):
        question = example.get("question") or example.get("instruction")  # Spider uses "question"
        sql_query = example.get("query") or example.get("sql")            # Spider uses "query" 
        db_id = example["db_id"]
        
        # Check if this db_id matches any available database
        if db_id in available_db_dirs:
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
            
            # If the exact file doesn't exist, try to find an alternative sqlite file
            if not os.path.isfile(db_path):
                db_files = [f for f in os.listdir(os.path.join(db_dir, db_id)) 
                          if f.endswith('.sqlite')]
                if db_files:
                    db_path = os.path.join(db_dir, db_id, db_files[0])
                else:
                    skipped_count += 1
                    continue
            
            # Get schema and sample rows text for this database
            try:
                schema_text = get_schema_and_samples(db_path)
                
                # Format data according to the mapping function described in the article
                base_example = {
                    "schema_with_rows": schema_text,
                    "question": question.strip(),
                    "db_id": db_id
                }
                
                # Add the original SQL query with T-style aliases
                has_aliases = " as t" in sql_query.lower()
                
                if has_aliases:
                    alias_stats["with_aliases"] += 1
                else:
                    alias_stats["without_aliases"] += 1
                
                # Always include the original query
                original_example = base_example.copy()
                original_example["query"] = sql_query.strip()
                processed_examples.append(original_example)
                
                # If the query uses T-style aliases, also add a simplified version
                if has_aliases and balance_aliases:
                    # Generate a simplified version without unnecessary aliases
                    simplified_sql = simplify_sql_query(sql_query, db_id)
                    
                    # Only add if it's different from the original
                    if simplified_sql != sql_query:
                        simple_example = base_example.copy()
                        simple_example["query"] = simplified_sql
                        processed_examples.append(simple_example)
                
                # For complex queries without aliases, optionally add a normalized version
                # with meaningful aliases to provide diverse learning examples
                elif not has_aliases and " join " in sql_query.lower() and balance_aliases:
                    normalized_sql = normalize_table_aliases(sql_query)
                    if normalized_sql != sql_query:
                        normalized_example = base_example.copy()
                        normalized_example["query"] = normalized_sql
                        processed_examples.append(normalized_example)
                
            except sqlite3.Error as e:
                skipped_count += 1
                continue
        else:
            skipped_count += 1
    
    print(f"Processed {len(processed_examples)} examples, skipped {skipped_count} examples")
    print(f"Original query distribution: {alias_stats['with_aliases']} with T-style aliases, {alias_stats['without_aliases']} without")
    
    # Create a dataset from the processed examples
    dataset = Dataset.from_list(processed_examples)
    
    # Apply the mapping function to format data for fine-tuning
    def _mapper(rec):
        schema = rec["schema_with_rows"].strip()
        question = rec["question"].strip()
        query = rec["query"].strip()
        db_id = rec.get("db_id", "")

        user_message = USER_MESSAGE_FORMAT.format(schema=schema, question=question)
        
        # Include system prompt and DB ID in the user message for Qwen format
        full_user_message = f"{SYSTEM_PROMPT}\n\nDatabase: {db_id}\n\n{user_message}"
        
        # Format for Qwen
        prompt = QWEN_CHAT_FORMAT.format(message=full_user_message, response=query)
        
        return {"text": prompt}
    
    # Transform the dataset with progress bar
    print("Formatting examples for Qwen...")
    formatted_dataset = dataset.map(_mapper)
    
    # Save the dataset to JSONL
    print(f"Saving {len(formatted_dataset)} examples to {output_path}...")
    formatted_dataset.to_json(output_path, orient="records", lines=True)
    print(f"Finished processing. Output saved to {output_path}")
    
    return formatted_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Spider dataset for fine-tuning with chat templates.")
    parser.add_argument("--train_file", type=str, default="spider_data/train_spider.json",
                        help="Path to Spider training JSON file (train_spider.json)")
    parser.add_argument("--db_dir", type=str, default=os.path.join(os.getcwd(), "spider_data/database"),
                        help="Path to the directory containing Spider databases (SQLite files)")
    parser.add_argument("--output_file", type=str, default="spider_finetune.jsonl",
                        help="Output path for the generated JSONL training file")
    parser.add_argument("--balance_aliases", action="store_true", default=True,
                        help="Balance examples with and without table aliases")
    args = parser.parse_args()
    
    process_spider_for_finetuning(args.train_file, args.db_dir, args.output_file, args.balance_aliases)
