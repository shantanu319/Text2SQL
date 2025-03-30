from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SchedulerType
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType  # for LoRA
from transformers import BitsAndBytesConfig  # Import directly from transformers
import os
import sys
import json
import re
import sqlite3  # Add this import for database access
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess
import torch.backends.cudnn as cudnn
import bitsandbytes as bnb
from transformers import DataCollatorForSeq2Seq

# needed for tokenization
def tokenize_function(examples, tokenizer, max_length=512):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    inputs = examples.get("input", [""] * len(examples["instruction"]))
    db_ids = examples.get("db_id", [""] * len(examples["instruction"]))
    ext_knowledge = examples.get("external_knowledge", [
                                 ""] * len(examples["instruction"]))

    for instruction, context, db_id, ext_know, response in zip(
        examples["instruction"], inputs, db_ids, ext_knowledge, examples["output"]
    ):
        # Format the prompt with db_id and external knowledge
        db_info = f"Database: {db_id}" if db_id else ""
        knowledge_info = f"External Knowledge: {ext_know}" if ext_know else ""
        context_info = f"{context}" if context else ""

        # Combine all context information
        full_context = "\n\n".join(
            filter(None, [db_info, knowledge_info, context_info]))

        prompt = f"<im_start>user\n{instruction}\n\n{full_context}<im_end>\n<im_start>assistant\n{response}<im_end>"
        prompts.append(prompt)

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized["attention_mask"] = [
        [1 if token_id != tokenizer.pad_token_id else 0 for token_id in seq]
        for seq in tokenized["input_ids"]
    ]
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# splits up and tokenizes query templates
def prepare_sql_dataset(file_path, tokenizer, test_size=0.4, max_length=512):
    templates = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    template = json.loads(line)
                    templates.append(template)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue

    dataset_dict = {"instruction": [], "input": [], "output": [], "db_id": [], "external_knowledge": []}
    for template in templates:
        # Check for different possible field names
        instruction = template.get("instruction", template.get("question", template.get("query", "")))
        if not instruction:
            print(f"Warning: Skipping template without instruction/question/query: {template}")
            continue
            
        output = template.get("output", template.get("query", template.get("sql", "")))
        if not output:
            print(f"Warning: Skipping template without output/query/sql: {template}")
            continue
            
        dataset_dict["instruction"].append(instruction)
        dataset_dict["input"].append(template.get("input", ""))
        dataset_dict["output"].append(output)
        dataset_dict["db_id"].append(template.get("db_id", ""))
        dataset_dict["external_knowledge"].append(template.get("external_knowledge", ""))

    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=test_size, seed=42)

    tokenized_train = dataset["train"].map(lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True, remove_columns=["instruction", "input","output", "db_id", "external_knowledge"])
    tokenized_val = dataset["test"].map(lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True, remove_columns=["instruction", "input","output", "db_id", "external_knowledge"])
    
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val


def load_spider_dataset(spider_dir, tokenizer, max_length=512, test_size=0.4):
    spider_files = []
    for root, dirs, files in os.walk(spider_dir):
        for file in files:
            if file.endswith('.jsonl') and ('spider' in file.lower() or 'eval' in file.lower()):
                spider_files.append(os.path.join(root, file))

    if not spider_files:
        return None, None

    all_examples = []
    for file_path in spider_files:
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                question = example.get(
                    'question', example.get('instruction', ''))
                if question:
                    all_examples.append({
                        "instruction": question,
                        "input": "",
                        "output": "",
                        "db_id": "",
                        "external_knowledge": ""
                    })

    if not all_examples:
        return None, None

    spider_dataset = Dataset.from_list(all_examples)

    train_df, val_df = train_test_split(
        pd.DataFrame(spider_dataset), test_size=test_size, random_state=42
    )
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input",
                        "output", "db_id", "external_knowledge"]
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input",
                        "output", "db_id", "external_knowledge"]
    )
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val

# finetunes model using LoRA, lora_r = rank of the low-rank adaptation, lora_alpha = scaling factor for LoRA
def finetune_model_lora(tokenizer, dataset_train, dataset_val, model_name, output_dir, batch_size=4, learning_rate=1e-4, num_train_epochs=3, gradient_accumulation_steps=8, fp16=True, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """Fine-tune a model with LoRA to generate SQL queries."""
    print("Starting QLoRA fine-tuning")
    
    # Check if CUDA is available and print info
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU for training.")
        device = torch.device("cpu")
    
    # Load model with LoRA configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        device_map="auto",  # Let accelerate handle device mapping
        trust_remote_code=True
    )
    
    # Move all parameters to the same device before creating LoRA layers
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "o_proj", "q_proj", "fc_in", "k_proj", "fc_out",
            "query_key_value", "down_proj", "v_proj", "up_proj", "gate_proj"
        ]
    )
    
    # Prepare model with LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",
        fp16=fp16,
    )
    
    # Setup data collator for batching sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=data_collator,
    )
    
    # Train the model
    print(f"Training on {len(dataset_train)} examples, evaluating on {len(dataset_val)} examples")
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save the model config with model_type information
    try:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            # Get the base model config
            base_model_config = model.config.to_dict()
            # Ensure model_type is present
            if "model_type" not in base_model_config:
                base_model_config["model_type"] = "qwen2"  # Set appropriate model type
            json.dump(base_model_config, f)
        print("Saved model config with model_type information")
    except Exception as e:
        print(f"Warning: Failed to save model config: {e}")
    
    # Evaluate after training
    print("Evaluating final model")
    metrics = trainer.evaluate()
    
    return model, metrics

def load_templates(file_path):
    templates = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    template = json.loads(line)
                    templates.append(template)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
    return templates

def generate_sql_from_prompt(model, tokenizer, prompt, schema_info=None, db_id=None, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=512):
    """Generate a SQL query from a prompt."""
    # Construct a more detailed prompt with schema information if available
    full_prompt = f"<im_start>system\nYou are a text-to-SQL assistant. Generate syntactically correct SQL queries based on the provided database schema (which includes column types and sample rows). Please follow these guidelines:\n- Use only tables and columns from the schema\n- Avoid unnecessary joins and table aliases when possible\n- Use simple direct queries where possible\n- Do not output any extra commentary\n<im_end>\n"
    
#     # Add in-context learning examples to help the model understand the task
#     icl_examples = """<im_start>user
# Database: university_system

# Database Schema:
# CREATE TABLE "Campuses" (
# 	"Id" INTEGER PRIMARY KEY,
# 	"Campus" TEXT,
# 	"Location" TEXT,
# 	"County" TEXT,
# 	"Year" INTEGER
# );
# /*
# Id    Campus    Location    County    Year
# 1    California State University-Bakersfield    Bakersfield    Kern    1965
# 2    California State University-Channel Islands    Camarillo    Ventura    2002
# 3    California State University-Chico    Chico    Butte    1887
# */

# CREATE TABLE "csu_fees" (
# 	"Campus" INTEGER PRIMARY KEY,
# 	"Year" INTEGER,
# 	"CampusFee" INTEGER,
# 	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
# );
# /*
# Campus    Year    CampusFee
# 1    1996    1951
# 2    2003    1868
# 3    1996    2042
# */

# CREATE TABLE "degrees" (
# 	"Year" INTEGER,
# 	"Campus" INTEGER,
# 	"Degrees" INTEGER,
# 	PRIMARY KEY (Year, Campus),
# 	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
# );
# /*
# Year    Campus    Degrees
# 1990    1    701
# 1991    1    681
# 1992    1    791
# */

# CREATE TABLE "discipline_enrollments" (
# 	"Campus" INTEGER,
# 	"Discipline" INTEGER,
# 	"Year" INTEGER,
# 	"Undergraduate" INTEGER,
# 	"Graduate" INTEGER,
# 	PRIMARY KEY (Campus, Discipline),
# 	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
# );
# /*
# Campus    Discipline    Year    Undergraduate    Graduate
# 1    4    2004    248    0
# 1    5    2004    811    73
# 1    6    2004    199    0
# */

# CREATE TABLE "enrollments" (
# 	"Campus" INTEGER,
# 	"Year" INTEGER,
# 	"TotalEnrollment_AY" INTEGER,
# 	"FTE_AY" INTEGER,
# 	PRIMARY KEY(Campus, Year),
# 	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
# );
# /*
# Campus    Year    TotalEnrollment_AY    FTE_AY
# 1    1956    384    123
# 1    1957    432    151
# 1    1958    422    178
# */

# CREATE TABLE "faculty" (
# 	"Campus" INTEGER,
# 	"Year" INTEGER,
# 	"Faculty" REAL,
# 	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
# );
# /*
# Campus    Year    Faculty
# 1    2002    357.1
# 2    2002    48.4
# 3    2002    742.8
# */

# Find the name of the campuses that is in Northridge, Los Angeles or in San Francisco, San Francisco.
# <im_end>

# <im_start>assistant
# SELECT Campus FROM Campuses WHERE Location="Northridge" AND County="Los Angeles" UNION SELECT Campus FROM Campuses WHERE Location="San Francisco" AND County="San Francisco";
# <im_end>

# <im_start>user
# Database: allergies

# Database Schema:
# CREATE TABLE Allergy_Type (
#        Allergy 		  VARCHAR(20) PRIMARY KEY,
#        AllergyType 	  VARCHAR(20)
# );

# CREATE TABLE Has_Allergy (
#        StuID 		 INTEGER,
#        Allergy 		 VARCHAR(20),
#        FOREIGN KEY(StuID) REFERENCES Student(StuID),
#        FOREIGN KEY(Allergy) REFERENCES Allergy_Type(Allergy)
# );

# CREATE TABLE Student (
#         StuID        INTEGER PRIMARY KEY,
#         LName        VARCHAR(12),
#         Fname        VARCHAR(12),
#         Age      INTEGER,
#         Sex      VARCHAR(1),
#         Major        INTEGER,
#         Advisor      INTEGER,
#         city_code    VARCHAR(3)
#  );

# Which allergy type has most number of allergies?
# <im_end>

# <im_start>assistant
# SELECT AllergyType FROM Allergy_Type GROUP BY AllergyType ORDER BY count(*) DESC LIMIT 1;
# <im_end>

# <im_start>user
# Database: research_institutions

# Database Schema:
# CREATE TABLE "building" (
# "building_id" text,
# "Name" text,
# "Street_address" text,
# "Years_as_tallest" text,
# "Height_feet" int,
# "Floors" int,
# PRIMARY KEY("building_id")
# );

# CREATE TABLE "Institution" (
# "Institution_id"  text,
# "Institution" text,
# "Location" text,
# "Founded" real,
# "Type" text,
# "Enrollment" int,
# "Team" text,
# "Primary_Conference" text,
# "building_id" text,
# PRIMARY KEY("Institution_id"),
# FOREIGN  KEY ("building_id") REFERENCES "building"("building_id")
# );

# CREATE TABLE "protein" (
# "common_name" text,
# "protein_name" text,
# "divergence_from_human_lineage" real,
# "accession_number" text,
# "sequence_length" real,
# "sequence_identity_to_human_protein" text,
# "Institution_id" text,
# PRIMARY KEY("common_name"),
# FOREIGN KEY("Institution_id") REFERENCES "Institution"("Institution_id")
# );

# For each building, show the name of the building and the number of institutions in it.
# <im_end>

# <im_start>assistant
# SELECT T1.name, count(*) FROM building AS T1 JOIN Institution AS T2 ON T1.building_id=T2.building_id GROUP BY T1.building_id;
# <im_end>

# <im_start>user
# Database: music_stadium

# Database Schema:
# CREATE TABLE stadium (
# Stadium_ID int,
# Location text,
# Name text,
# Capacity int,
# Highest int,
# Lowest int,
# Average int,
# PRIMARY KEY (Stadium_ID)
# )
# /*
# Stadium_ID   Location   Name   Capacity   Highest   Lowest   Average
# 1   Raith Rovers   Stark's Park   10104   4812   1294   2106
# 2   Ayr United   Somerset Park   11998   2363   1057   1477
# 3   East Fife   Bayview Stadium   2000   1980   533   864
# */

# CREATE TABLE singer (
# Singer_ID int,
# Name text,
# Country text,
# Song_Name text,
# Song_release_year text,
# Age int,
# Is_male bool,
# PRIMARY KEY (Singer_ID)
# )
# /*
# Singer_ID    Name    Country    Song_Name    Song_release_year    Age    Is_male
# 1    Joe Sharp    Netherlands    You    1992    52    F
# 2    Timbaland    United States    Dangerous    2008    32    T
# 3    Justin Brown    France    Hey Oh    2013    29    T
# */

# CREATE TABLE concert (
# concert_ID int,
# concert_Name text,
# Theme text,
# Stadium_ID text,
# Year text,
# PRIMARY KEY (concert_ID),
# FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID)
# )
# /*
# concert_ID    concert_Name    Theme    Stadium_ID    Year
# 1    Auditions    Free choice    1    2014
# 2    Super bootcamp    Free choice 2    2    2014
# 3    Home Visits    Bleeding Love    2    2015
# */

# CREATE TABLE singer_in_concert (
# concert_ID int,
# Singer_ID text,
# PRIMARY KEY (concert_ID,Singer_ID),
# FOREIGN KEY (concert_ID) REFERENCES concert(concert_ID),
# FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
# )
# /*
# concert_ID    Singer_ID
# 1    2
# 1    3
# 1    5
# */
# <im_end>
# """
    
    # # Include the ICL examples in the full prompt
    # full_prompt += icl_examples
    
    if schema_info:
        # Include detailed schema information in the prompt
        schema_prompt = f"<im_start>user\nDatabase: {db_id}\n\nDatabase Schema:\n{schema_info}\n\nNow, please convert the following question to a SQL query:\n{prompt}\n<im_end>\n"
    else:
        # Fallback to a simpler prompt without schema details
        schema_prompt = f"<im_start>user\nDatabase: {db_id}\n\nPlease convert the following question to a SQL query:\n{prompt}\n<im_end>\n"
    
    full_prompt += schema_prompt
    
    # Check if tokens exceed max length and truncate if necessary
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] > 3800:  # Safety margin for max length
        print(f"Warning: Input too long ({inputs.input_ids.shape[1]} tokens), truncating prompt.")
        inputs = tokenizer(full_prompt, truncation=True, max_length=3800, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,  # Enable sampling for more diverse outputs
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the SQL query
    if "<im_end>" in response and "<im_start>assistant" in response:
        parts = response.split("<im_start>assistant\n")
        if len(parts) > 1:
            sql_query = parts[1].split("<im_end>")[0].strip()
        else:
            sql_query = ""
    else:
        # Try to extract just the SQL part using pattern matching
        assistant_idx = response.find("assistant")
        if assistant_idx != -1:
            sql_query = response[assistant_idx:].replace("assistant", "", 1).strip()
        else:
            sql_query = response.replace(full_prompt, "").strip()
    
    # Clean up the response - remove any sql code blocks or formatting
    sql_query = clean_sql_response(sql_query)
    
    # Further normalize to remove unnecessary aliases and trailing characters
    sql_query = normalize_sql(sql_query)
    
    return sql_query

def clean_sql_response(response):
    """Clean up the SQL response to extract just the SQL query."""
    # For Qwen models which output in the format <im_start>response SQL_QUERY <im_end>
    if "<im_start>response" in response and "<im_end>" in response:
        # Extract content between the response tag and im_end
        try:
            response = response.split("<im_start>response")[1].split("<im_end>")[0].strip()
            # If there's a <sep> tag, take only the content before it (the SQL part)
            if "<sep>" in response:
                response = response.split("<sep>")[0].strip()
        except:
            pass  # If the splitting fails, continue with other methods
    
    # Remove SQL code block formatting if present
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    # Remove any trailing/leading punctuation or whitespace
    response = response.strip("\n\r\t ;")
    
    return response.strip()

def normalize_sql(sql):
    """Normalize SQL query to remove unnecessary elements (all the random t1. s and the t2. s)"""
    if not sql:
        return sql
    
    sql_lower = sql.lower()
    
    # Fix unnecessary table aliases in simple queries
    if " as t1" in sql_lower and " join " not in sql_lower:
        sql = re.sub(r'\s+as\s+t\d', '', sql, flags=re.IGNORECASE)
    
    # Remove trailing semicolons and adjust spacing
    sql = sql.strip().rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    
    # Remove unnecessary aliases where one table is involved
    # Replace t1.column with just column when there's no JOIN
    if " join " not in sql_lower:
        # This is a simple query with one table
        sql = re.sub(r't\d\.([a-zA-Z0-9_]+)', r'\1', sql, flags=re.IGNORECASE)
        
    # Replace any excessive whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql

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

def process_spider(spider_path, db_path, output_file, model_name=None, model_path=None):
    """Process the Spider dataset to generate SQL queries and evaluate."""
    if not model_path and not model_name:
        raise ValueError("Either model_name or model_path must be provided")
    
    # Load model and tokenizer
    if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print(f"Loading fine-tuned model with LoRA adapters from {model_path}")
        # Load the tokenizer from the base model
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Default base model
        
        # Try to get the base model from the adapter config
        try:
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
                if "base_model_name_or_path" in adapter_config:
                    base_model_name = adapter_config["base_model_name_or_path"]
                    print(f"Using base model: {base_model_name}")
        except Exception as e:
            print(f"Warning: Could not load adapter_config.json: {e}")
            print(f"Using default base model: {base_model_name}")
        
        # Load tokenizer from base model name
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Load PEFT model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Just load the model directly from model_name or model_path (no LoRA adapters)
        model_to_load = model_path if model_path else model_name
        print(f"Loading model {model_to_load} directly (no LoRA adapters)")
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_file, exist_ok=True)

    # Path to the original Spider dataset files - check multiple possible locations
    possible_paths = [
        os.path.join(spider_path, "spider", "dev.json"),
        os.path.join(spider_path, "dev.json"),
        os.path.join(spider_path, "spider", "evaluation_examples", "examples", "dev.json"),
        os.path.join(spider_path, "test-suite-sql-eval", "database", "dev.json"),
        os.path.join(spider_path, "Spider2", "dev.json")
    ]

    spider_dev_file = None
    for path in possible_paths:
        if os.path.exists(path):
            spider_dev_file = path
            print(f"Found Spider dev dataset at {spider_dev_file}")
            break

    if spider_dev_file is None:
        print("Spider dev dataset not found in any of the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Please ensure the Spider dataset is correctly installed.")
        return None

    # Load examples from Spider dev dataset
    examples = []
    try:
        with open(spider_dev_file, 'r') as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} examples from Spider dev dataset")

    except Exception as e:
        print(f"Error loading Spider dev dataset: {e}")
        return None
    
    # cut output for debugging if needed
    debug_limit = 500  # Set to None for full run
    if debug_limit is not None:
        print(f"Limiting evaluation to {debug_limit} examples (from {len(examples)} total)")
        examples = examples[:debug_limit]

    # Preload database schemas for faster processing
    db_schemas = {}
    print("Loading database schemas...")
    for example in tqdm(examples, desc="Loading schemas"):
        db_id = example.get("db_id", "")
        if db_id and db_id not in db_schemas:
            # Try to locate the database file
            db_file_path = find_database_path(db_path, db_id)
            if db_file_path:
                try:
                    db_schemas[db_id] = get_schema_and_samples(db_file_path)
                except Exception as e:
                    print(f"Error loading schema for {db_id}: {e}")
                    db_schemas[db_id] = None
    
    print(f"Loaded schemas for {len(db_schemas)} databases")
    
    # Create output files
    pred_file_path = os.path.join(output_file, "pred_spider.sql")
    gold_file_path = os.path.join(output_file, "gold_spider.sql")

    # Process examples in batches for better progress tracking
    predictions = []
    gold_queries = []
    detailed_results = []

    for i, example in enumerate(tqdm(examples, desc="Generating SQL queries")):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold_query = example.get("query", "")

        # Skip if no question or database ID
        if not question or not db_id:
            print(f"Skipping example {i}: Missing question or database ID")
            continue

        # Get schema information for this database
        schema_info = db_schemas.get(db_id)
        
        # Generate SQL query with schema information
        generated_sql = generate_sql_from_prompt(
            model,
            tokenizer,
            question,
            schema_info=schema_info,
            db_id=db_id,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_new_tokens=512
        )

        # Store prediction and gold query
        predictions.append(generated_sql)
        gold_queries.append(f"{gold_query}\t{db_id}")
        
        # Store detailed result for analysis
        detailed_results.append({
            "db_id": db_id,
            "question": question,
            "gold_query": gold_query,
            "predicted_query": generated_sql
        })

    # Write predictions to file
    with open(pred_file_path, 'w') as f:
        for sql in predictions:
            f.write(f"{sql}\n")

    # Write gold queries to file
    with open(gold_file_path, 'w') as f:
        for gold in gold_queries:
            f.write(f"{gold}\n")

    print(f"Generated SQL queries written to {pred_file_path}")
    print(f"Gold SQL queries written to {gold_file_path}")
    
    # Save detailed results for analysis
    detailed_results_file = os.path.join(output_file, "detailed_results.json")
    with open(detailed_results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Detailed results saved to {detailed_results_file}")

    # Try to run the Spider evaluation script
    eval_script = os.path.join(spider_path, "spider", "evaluation.py")
    if not os.path.exists(eval_script):
        eval_script = os.path.join(spider_path, "evaluation.py")
    tables_json = os.path.join(spider_path, "spider", "tables.json")
    if not os.path.exists(tables_json):
        tables_json = os.path.join(spider_path, "tables.json")
    
    if not os.path.exists(eval_script):
        print(f"Spider evaluation script not found at {eval_script}")
        return {"predictions": predictions, "gold": gold_queries, "detailed_results": detailed_results}

    if not os.path.exists(tables_json):
        print(f"Spider tables.json not found at {tables_json}")
        return {"predictions": predictions, "gold": gold_queries, "detailed_results": detailed_results}

    print(f"Running Spider evaluation with script at {eval_script}")

    try:
        cmd = [
            sys.executable,
            eval_script,
            "--gold", gold_file_path,
            "--pred", pred_file_path,
            "--etype", "all",  # Evaluate all types of queries
            "--table", tables_json,  # Path to tables.json
            "--db", os.path.join(db_path, "database")  # Path to database files
        ]

        print("Executing command:", " ".join(cmd))
        completed_process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print("Evaluation completed successfully.")
        print(completed_process.stdout)

        # Parse evaluation results
        eval_metrics = {}
        for line in completed_process.stdout.split('\n'):
            if "exact matching accuracy:" in line:
                eval_metrics["exact_match"] = float(line.split(":")[1].strip())
            elif "execution accuracy:" in line:
                eval_metrics["execution"] = float(line.split(":")[1].strip())
            elif "partial matching accuracy:" in line:
                eval_metrics["partial_match"] = float(
                    line.split(":")[1].strip())

        print("Evaluation Metrics:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")

        results = {
            "predictions": predictions,
            "gold": gold_queries,
            "evaluation_metrics": eval_metrics,
            "detailed_results": detailed_results
        }

        # Save results to JSON file
        results_file = os.path.join(output_file, "spider_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation results saved to {results_file}")
        return results

    except Exception as e:
        print(f"Error running evaluation script: {e}")
        if hasattr(e, 'stderr'):
            print(e.stderr)

        results = {
            "predictions": predictions,
            "gold": gold_queries,
            "error": str(e),
            "detailed_results": detailed_results
        }

        # Save results even if evaluation failed
        results_file = os.path.join(output_file, "spider_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file} (without evaluation metrics)")
        return results

def find_database_path(db_dir, db_id):
    """Find the path to a database file given its ID."""
    # First try direct path
    db_path = os.path.join(db_dir, "database", db_id, f"{db_id}.sqlite")
    if os.path.isfile(db_path):
        return db_path
    
    # Try subdirectory structure 
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
    if os.path.isfile(db_path):
        return db_path
    
    # Look for any sqlite file in the directory
    potential_dirs = [
        os.path.join(db_dir, "database", db_id),
        os.path.join(db_dir, db_id)
    ]
    
    for dir_path in potential_dirs:
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".sqlite") or filename.endswith(".db"):
                    return os.path.join(dir_path, filename)
    
    print(f"Could not find database file for {db_id}")
    return None

def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):
    if num_samples > len(test_templates):
        num_samples = len(test_templates)

    samples = random.sample(test_templates, num_samples)
    results = []

    for i, template in enumerate(samples):
        prompt = f"{template.get('instruction', '')}"
        if template.get('input', ''):
            prompt += f"\n\n{template.get('input', '')}"

        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)
        print(f"Sample #{i+1}")
        print(f"Instruction: {template.get('instruction', '')}")
        if template.get('input', ''):
            print(f"Input: {template.get('input', '')}")
        print(f"Expected SQL: {template.get('output', '')}")
        print(f"Generated SQL: {generated_sql}")

        is_correct = normalize_sql(template.get(
            "output", "")) == normalize_sql(generated_sql)
        print(f"Match: {'✓' if is_correct else '✗'}")
        print("-" * 50)

        results.append({
            "prompt": prompt,
            "expected_sql": template.get('output', ''),
            "generated_sql": generated_sql,
            "is_correct": is_correct
        })

    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = 100 * correct_count / len(results) if results else 0
    print(
        f"Overall Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")

    return results

def main():
    action = sys.argv[1] 
    data_file = "balanced_spider_finetune.jsonl" #finetune data
    spider_dir = os.path.expanduser("~/Documents/GitHub/txt2sql_461") # replace this with the parent directory of spider
    output_dir = "./fine_tuned_model"  # Output directory for finetuned models
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_model_name = "Qwen/Qwen-7B"  # Base Qwen 7B model

    if action == "finetune":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset, eval_dataset = prepare_sql_dataset(
            data_file, tokenizer, max_length=384)
        print(
            f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")

        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for training.")
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            print("MPS (Apple Silicon) is available. Using MPS for training.")
            torch.mps.empty_cache()
        else:
            print("No GPU detected. Using CPU for training (this will be slow).")

        model, metrics = finetune_model_lora(
            tokenizer=tokenizer,
            dataset_train=train_dataset,
            dataset_val=eval_dataset,
            model_name=model_name,
            output_dir=output_dir,
            batch_size=4,
            learning_rate=1e-4,
            num_train_epochs=2,
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        print(
            f"Fine-tuning completed! Loss: {metrics.get('eval_loss', 'N/A'):.2f}")

    elif action == "evaluate":
        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
        test_templates = load_templates(data_file)
        evaluate_finetuned_model(model, tokenizer, test_templates)

    elif action == "process_spider":
        torch.backends.cudnn.benchmark = True
        
        # By default, use the non-finetuned Qwen2.5-7B-Instruct
        target_model = model_name
        output_folder = "qwen25_instruct_spider_results"
        
        # But allow using the base Qwen 7B model as an option
        if len(sys.argv) > 2:
            if sys.argv[2] == "base":
                target_model = base_model_name
                output_folder = "base_qwen7b_spider_results"
                print(f"Using base Qwen 7B model: {target_model}")
            elif sys.argv[2] == "finetuned":
                target_model = output_dir  # Use the fine-tuned model path
                output_folder = "finetuned_spider_results"
                print(f"Using fine-tuned model from: {target_model}")
            else:
                print(f"Unknown model option: {sys.argv[2]}, using default Qwen2.5-7B-Instruct")
        
        print(f"Using model: {target_model}")
        print(f"Results will be saved to: {output_folder}")
        
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(
            target_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Number of available GPUs: {num_gpus}")
        
        # If we're using the fine-tuned model path, set model_path, otherwise use model_name
        model_path = None
        model_name_param = target_model
        
        if target_model == output_dir:
            model_path = target_model
            model_name_param = None
            
        # Process Spider dataset
        results = process_spider(
            spider_path="spider_data",
            db_path="spider_data", 
            output_file=output_folder,
            model_name=model_name_param,
            model_path=model_path
        )
        print(f"Processed {len(results['predictions'])} examples")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: finetune, evaluate, process_spider")
        print("To use base Qwen 7B: python qwen_setup.py process_spider base")
        print("To use finetuned model: python qwen_setup.py process_spider finetuned")
        print("To use default Qwen2.5-7B-Instruct: python qwen_setup.py process_spider")


if __name__ == "__main__":
    main()