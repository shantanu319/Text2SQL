import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import requests
from torch import nn
import torch.nn.functional as F
import zipfile
import io
import sqlite3

def setup_qwen():
    model_name = "Qwen/Qwen-32B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

class ReFoRCe(nn.Module):
    def __init__(self, base_model, tokenizer, feedback_steps=3):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.feedback_steps = feedback_steps
        
    def get_feedback(self, output_text):
        return "Consider reviewing your answer for factual accuracy and logical consistency."
    
    def forward(self, input_text):
        for i in range(self.feedback_steps):
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.base_model.device)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if i < self.feedback_steps - 1:  
                feedback = self.get_feedback(output_text)
                
                input_text = f"{output_text}\n\nFeedback: {feedback}\n\nImproved answer:"
            
        return output_text

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

def example_usage():
    model, tokenizer = setup_qwen()
    
    reforce_model = ReFoRCe(model, tokenizer)
    
    prompt = "Explain the concept of quantum computing in simple terms."
    
    print("Generating with Qwen-32B directly:")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print("\nGenerating with ReFoRCe (iterative feedback):")
    response = reforce_model(prompt)
    print(response)

if __name__ == "__main__":
    run_on_spider_dataset(model_type="both", num_examples=10)
