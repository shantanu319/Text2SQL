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

def setup_qwen():
    model_name = "Qwen/QwQ-32B"  # Use Qwen/QwQ-32B model
    
    # Load tokenizer with proper chat template settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"  # Important for proper padding
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        low_cpu_mem_usage=True       # Reduce memory usage
    )
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=1024):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        context = examples["input"][i]
        response = examples["output"][i]

        prompt = f"{tokenizer.bos_token}user\n{instruction}\n\n{context}\n{tokenizer.eos_token}assistant\n{response}{tokenizer.eos_token}"
        prompts.append(prompt)
    
    tokenized_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    # attn mask
    tokenized_inputs["attention_mask"] = (tokenized_inputs["input_ids"] != tokenizer.pad_token_id).long()
    
    return tokenized_inputs

def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=1024):
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
                  batch_size=1, learning_rate=2e-5, num_train_epochs=3, 
                  gradient_accumulation_steps=8, fp16=True):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for large models
        low_cpu_mem_usage=True,      # Reduce memory usage
        device_map="auto",           # Automatically distribute across available devices
        trust_remote_code=True
    )
 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
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
        # Add more memory optimizations
        gradient_checkpointing=True,
        optim="adamw_torch",
        # Required for avoiding padding issues with non-even batch sizes
        dataloader_drop_last=True,
        # Prevent NaN loss issues
        no_cuda=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate the model
    eval_metrics = trainer.evaluate()
    
    return model, eval_metrics

def load_templates(file_path):
    with open(file_path, 'r') as f:
        templates = json.load(f)
    return templates

def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=512):

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Format prompt using Qwen's chat format
    formatted_prompt = f"{tokenizer.bos_token}user\n{prompt}\n\n{tokenizer.eos_token}assistant\n"
    
    # Tokenize prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate SQL
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode generated SQL and clean up
    generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean up the generated text to extract just the SQL
    generated_sql = generated_text.strip()
    
    return generated_sql

def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):

    if num_samples > len(test_templates):
        num_samples = len(test_templates)
    
    samples = random.sample(test_templates, num_samples)
    
    results = []
    
    for template in samples:
        # Create prompt
        prompt = f"{template['instruction']}\n\n{template['input']}"
        
        # Generate SQL
        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)
        
        # Add result
        results.append({
            "prompt": prompt,
            "expected_sql": template["output"],
            "generated_sql": generated_sql
        })
    
    return results

def normalize_sql(sql):
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    
    # Remove trailing semicolon
    if sql.endswith(';'):
        sql = sql[:-1]
    
    # Convert to lowercase
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
    # Parse command-line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "finetune"
    
    # Set up paths
    data_file = "sql_templates.json"
    output_dir = "./fine_tuned_model"
    
    # Model name or path - use a tiny model that's easier to fine-tune
    model_name = "Qwen/QwQ-32B"  # Use Qwen/QwQ-32B model
    
    if action == "finetune":
        print("Starting fine-tuning process...")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        print("Preparing dataset...")
        train_dataset, eval_dataset = prepare_sql_dataset(data_file, tokenizer)
        
        print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")
        
        # Fine-tune model
        print("Fine-tuning model...")
        model, metrics = finetune_model(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=1,
            learning_rate=2e-5,
            num_train_epochs=3,
            fp16=True  # Enable mixed precision for compatibility
        )
        
        print("Fine-tuning completed!")
        print(f"Evaluation metrics: {metrics}")
        
    elif action == "evaluate":
        print("Evaluating fine-tuned model...")
        
        # Load fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        # Load test templates
        test_templates = load_templates(data_file)
        
        # Evaluate model
        results = evaluate_finetuned_model(model, tokenizer, test_templates)
        
        # Print results
        print("\nEvaluation Results:")
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