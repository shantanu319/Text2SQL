# import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import requests
# from torch import nn
# import torch.nn.functional as F
import zipfile
import io
import sqlite3
import re
import sys
import random
from finetune_qwen import finetune_model, load_templates

"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/QwQ-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r's are in the word \"strawberry\""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
"""

def setup_qwen():
    model_name = "Qwen/QwQ-32B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def calculate_exact_match(results):
    """Calculate exact match accuracy after normalizing SQL queries"""
    matches = 0
    for result in results:
        gold_normalized = normalize_sql(result['gold_sql'])
        gen_normalized = normalize_sql(result['generated_sql'])
        if gold_normalized == gen_normalized:
            matches += 1
    return matches / len(results) if results else 0

def calculate_execution_accuracy(results, db_dir):
    """Calculate execution accuracy by running queries against databases"""
    matches = 0
    for result in results:
        db_path = os.path.join(db_dir, result['db_id'], result['db_id'] + '.sqlite')
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            # Execute gold query
            gold_result = execute_query(conn, result['gold_sql'])
            # Execute generated query
            gen_result = execute_query(conn, result['generated_sql'])
            # Compare results (ignoring order)
            if compare_results(gold_result, gen_result):
                matches += 1
            conn.close()
        except Exception as e:
            print(f"Error executing query: {e}")
    return matches / len(results) if results else 0

def download_spider_dataset():
    data_dir = "spider_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Downloading Spider 2.0 dataset...")
    
    # Download the dev set
    dev_url = "https://yale-lily.github.io/spider2/spider2-snow/resource/data/dev.json"
    dev_response = requests.get(dev_url)
    with open(os.path.join(data_dir, "dev.json"), "wb") as f:
        f.write(dev_response.content)
    
    # Download the database files
    db_url = "https://yale-lily.github.io/spider2/spider2-snow/resource/data/spider.zip"
    db_response = requests.get(db_url)
    with zipfile.ZipFile(io.BytesIO(db_response.content)) as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"Spider dataset downloaded to {data_dir}")
    return data_dir

def load_spider_examples(data_dir, limit=None):
    dev_file = os.path.join(data_dir, "dev.json")
    with open(dev_file, "r") as f:
        examples = json.load(f)
    
    if limit:
        examples = examples[:limit]
    
    return examples

def format_spider_prompt(example):
    db_id = example["db_id"]
    question = example["question"]
    
    prompt = f"""Generate a SQL query for the following question:
Database: {db_id}
Question: {question}

SQL:"""
    
    return prompt

def evaluate_sql_generation(model, tokenizer, examples):
    results = []
    
    for example in tqdm(examples, desc="Evaluating"):
        prompt = format_spider_prompt(example)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                temperature=0.1,
                do_sample=False
            )
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sql = generated_sql[len(prompt):].strip()
        
        results.append({
            "question": example["question"],
            "gold_sql": example["query"],
            "generated_sql": generated_sql,
            "db_id": example["db_id"]
        })
    
    return results

def run_on_spider_dataset(model_type="both", num_examples=10):
    print(f"Running evaluation on Spider dataset with model: {model_type}")
    
    data_dir = download_spider_dataset()
    examples = load_spider_examples(data_dir, limit=num_examples)
    
    print(f"Loaded {len(examples)} examples from Spider dataset")
    
    model, tokenizer = setup_qwen()
    
    if model_type == "qwen" or model_type == "both":
        print("\n--- Evaluating with Qwen model directly ---")
        qwen_results = evaluate_sql_generation(model, tokenizer, examples)
        
        print("\nSample results from Qwen model:")
        for i, result in enumerate(qwen_results[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {result['question']}")
            print(f"Gold SQL: {result['gold_sql']}")
            print(f"Generated SQL: {result['generated_sql']}")
    
    if model_type == "reforce" or model_type == "both":
        print("\n--- Evaluating with Qwen + ReFoRCe ---")
        reforce_model = ReFoRCe(model, tokenizer)
        
        reforce_results = []
        for example in tqdm(examples, desc="Evaluating ReFoRCe"):
            prompt = format_spider_prompt(example)
            generated_sql = reforce_model(prompt)
            generated_sql = generated_sql[len(prompt):].strip()
            
            reforce_results.append({
                "question": example["question"],
                "gold_sql": example["query"],
                "generated_sql": generated_sql,
                "db_id": example["db_id"]
            })
        
        print("\nSample results from Qwen + ReFoRCe:")
        for i, result in enumerate(reforce_results[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {result['question']}")
            print(f"Gold SQL: {result['gold_sql']}")
            print(f"Generated SQL: {result['generated_sql']}")
    
    return {
        "qwen_results": qwen_results if model_type == "qwen" or model_type == "both" else None,
        "reforce_results": reforce_results if model_type == "reforce" or model_type == "both" else None
    }

def main():
    
    
    # Set up the model and tokenizer
    print("Setting up the base Qwen model...")
    model, tokenizer = setup_qwen()
    
    # Load SQL templates
    templates_file = "sql_templates.json"
    templates = load_templates(templates_file)
    
    # Select a random template for testing
    test_template = random.choice(templates)
    test_prompt = f"{test_template['instruction']}\n\n{test_template['input']}"
    gold_sql = test_template['output']
    
    print("\n" + "="*80)
    print("TESTING BASE MODEL")
    print("="*80)
    print(f"Test prompt: {test_prompt}")
    print(f"Gold SQL: {gold_sql}")
    
    # Generate SQL with base model
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    base_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nBase model result: {base_result}")
    
    # Fine-tune the model
    print("\n" + "="*80)
    print("FINE-TUNING MODEL")
    print("="*80)
    output_dir = "fine_tuned_qwen_sql"
    
    # Run the fine-tuning process
    fine_tuned_model, fine_tuned_tokenizer = finetune_model(
        templates_file=templates_file,
        output_dir=output_dir
    )
    
    print("\n" + "="*80)
    print("TESTING FINE-TUNED MODEL")
    print("="*80)
    print(f"Test prompt: {test_prompt}")
    print(f"Gold SQL: {gold_sql}")
    
    # Generate SQL with fine-tuned model
    inputs = fine_tuned_tokenizer(test_prompt, return_tensors="pt").to(fine_tuned_model.device)
    with torch.no_grad():
        outputs = fine_tuned_model.generate(**inputs, max_length=200)
    fine_tuned_result = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nFine-tuned model result: {fine_tuned_result}")
    
    # Compare the results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Base model exact match: {normalize_sql(base_result) == normalize_sql(gold_sql)}")
    print(f"Fine-tuned model exact match: {normalize_sql(fine_tuned_result) == normalize_sql(gold_sql)}")
    
    # Run evaluation on more examples if needed
    if "--full-eval" in sys.argv:
        print("\n" + "="*80)
        print("RUNNING FULL EVALUATION")
        print("="*80)
        base_results = run_on_spider_dataset(model_type="base", num_examples=10, model=model, tokenizer=tokenizer)
        ft_results = run_on_spider_dataset(model_type="base", num_examples=10, model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)
        
        print(f"Base model EM accuracy: {base_results['accuracy']:.2f}%")
        print(f"Fine-tuned model EM accuracy: {ft_results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
