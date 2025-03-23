from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os
import json
import sys
import random
import numpy as np
from typing import Dict, List, Any
import sqlite3
import re

# def setup_qwen():
#     model_name = "Qwen/QwQ-32B"  # load model
    
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True
#     ).to(device)
    
#     return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        context = examples["input"][i]
        response = examples["output"][i]
    
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
        prompts.append(prompt)

    tokenized_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # needed for causal
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    # pulls from query templates json
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    dataset_dict = {
        "instruction": [],
        "input": [],
        "output": []
    }
    
    for template in templates:
        dataset_dict["instruction"].append(template["instruction"])
        dataset_dict["input"].append(template["input"])
        dataset_dict["output"].append(template["output"])

    dataset = Dataset.from_dict(dataset_dict)

    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    tokenized_train = dataset["train"].map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    tokenized_val = dataset["test"].map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    return tokenized_train, tokenized_val

def finetune_model(model_name, train_dataset, eval_dataset, output_dir="./fine_tuned_model", 
                  batch_size=2, learning_rate=5e-5, num_train_epochs=3, 
                  gradient_accumulation_steps=4, fp16=False):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
        low_cpu_mem_usage=True      
    ).to(device)
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.01,
        fp16=fp16,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_metrics = trainer.evaluate()
    
    return model, eval_metrics

def load_templates(file_path):
    with open(file_path, 'r') as f:
        templates = json.load(f)
    return templates

def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=256):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = model.device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated SQL
    generated_sql = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return generated_sql

def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):
    if num_samples > len(test_templates):
        num_samples = len(test_templates)
    
    samples = random.sample(test_templates, num_samples)
    
    results = []
    
    for template in samples:
        prompt = f"{template['instruction']}\n\n{template['input']}"

        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)

        results.append({
            "prompt": prompt,
            "expected_sql": template["output"],
            "generated_sql": generated_sql
        })
    
    return results

def normalize_sql(sql): 
    sql = ' '.join(sql.split())
    if sql.endswith(';'):
        sql = sql[:-1]
    sql = sql.lower()
    
    return sql

def execute_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def compare_results(result1, result2):
    set1 = set(tuple(row) for row in result1)
    set2 = set(tuple(row) for row in result2)
    
    return set1 == set2

def main():
    torch.cuda.empty_cache()
    
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "finetune"  # Default action
        
    data_file = "sql_templates.json"
    output_dir = "./fine_tuned_model"
 
    model_name = "Qwen/Qwen-7B"  
    
    if action == "finetune":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # tokenize datasets
        train_dataset, eval_dataset = prepare_sql_dataset(data_file, tokenizer)
        
        print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")

        model, metrics = finetune_model(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=1,
            learning_rate=5e-5,
            num_train_epochs=3,
            fp16=True  # Disable mixed precision for compatibility
        ).to(device)
        
        print("Fine-tuning completed!")
        print(f"Evaluation metrics: {metrics}")
        
    elif action == "evaluate":
    
        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        test_templates = load_templates(data_file)
        
        results = evaluate_finetuned_model(model, tokenizer, test_templates)

        print("\nExact match results:")
        for i, result in enumerate(results):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {result['prompt']}")
            print(f"Expected SQL: {result['expected_sql']}")
            print(f"Generated SQL: {result['generated_sql']}")
            
            # Check if correct
            if normalize_sql(result['expected_sql']) == normalize_sql(result['generated_sql']):
                print(" Correct")
            else:
                print(" Incorrect")
    
    else:
        print(f"Unknown action: {action}")
        print("Usage: python qwen_setup.py [finetune|evaluate]")

if __name__ == "__main__":
    main()