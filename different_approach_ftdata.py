import os
import json
import sqlite3
import argparse

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

def process_spider_to_jsonl(train_json_path, db_dir, output_path):
    # Load Spider training examples
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
        
    # Get list of available databases
    available_db_dirs = [d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))]
    print(f"Found {len(available_db_dirs)} available database directories")
    
    # Count of examples processed
    processed_count = 0
    skipped_count = 0
    
    # Open output file for writing JSONL
    with open(output_path, 'w') as out_f:
        for example in train_data:
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
                        print(f"Using alternative database file for {db_id}: {db_path}")
                    else:
                        print(f"Warning: No SQLite files found in directory for {db_id}")
                        skipped_count += 1
                        continue
                
                # Get schema and sample rows text for this database
                try:
                    schema_text = get_schema_and_samples(db_path)
                    # Build the input context: include database name, and the schema with sample data
                    db_header = f"Database: {db_id}"
                    full_context = f"{db_header}\nSchema:\n{schema_text}"
                    # Create the JSON record
                    record = {
                        "instruction": question.strip(),
                        "input": full_context,
                        "output": sql_query.strip()
                    }
                    # Write as a single line JSON
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1
                except sqlite3.Error as e:
                    print(f"Error processing database {db_id}: {e}")
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
                # Uncomment to see which databases are missing - this will produce a lot of output
                # print(f"Skipping example with unavailable database: {db_id}")
        
    print(f"Processed {processed_count} examples, skipped {skipped_count} examples")
    print(f"Finished processing. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Spider dataset for Qwen fine-tuning.")
    parser.add_argument("--train_file", type=str, default="spider/evaluation_examples/examples/train_spider.json",
                        help="Path to Spider training JSON file (train_spider.json)")
    parser.add_argument("--db_dir", type=str, default="spider_data/database",
                        help="Path to the directory containing Spider databases (SQLite files)")
    parser.add_argument("--output_file", type=str, default="spider_finetune.jsonl",
                        help="Output path for the generated JSONL training file")
    args = parser.parse_args()
    
    process_spider_to_jsonl(args.train_file, args.db_dir, args.output_file)
