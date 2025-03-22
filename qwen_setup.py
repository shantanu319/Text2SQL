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
    model_name = "gpt2"  # Use a small model that will definitely work locally
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the examples for fine-tuning.
    
    Args:
        examples: Dictionary containing the examples
        tokenizer: Tokenizer to use for tokenization
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Combine instruction, input, and output into a single text
    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        context = examples["input"][i]
        response = examples["output"][i]
        
        # Create formatted prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
        prompts.append(prompt)
    
    # Tokenize the prompts
    tokenized_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Set the labels to the input_ids (for causal language modeling)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    """
    Prepare the SQL dataset for fine-tuning.
    
    Args:
        file_path: Path to the JSON file containing the SQL templates
        tokenizer: Tokenizer to use for tokenization
        test_size: Fraction of data to use for validation
        max_length: Maximum sequence length
        
    Returns:
        Train and validation datasets
    """
    # Load the SQL templates
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    # Convert to dataset format
    dataset_dict = {
        "instruction": [],
        "input": [],
        "output": []
    }
    
    for template in templates:
        dataset_dict["instruction"].append(template["instruction"])
        dataset_dict["input"].append(template["input"])
        dataset_dict["output"].append(template["output"])
    
    # Create Dataset object
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    # Tokenize the datasets
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
    """
    Fine-tune the model on the SQL dataset.
    
    Args:
        model_name: Name or path of the model to fine-tune
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save the fine-tuned model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use mixed precision training
        
    Returns:
        Fine-tuned model and training metrics
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if it's not already set (needed for GPT-2 and similar models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load a smaller model for fine-tuning with reduced precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 instead of default
        low_cpu_mem_usage=True      # Reduce memory usage
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Set up training arguments
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
    """
    Load SQL templates from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the SQL templates
        
    Returns:
        List of templates
    """
    with open(file_path, 'r') as f:
        templates = json.load(f)
    return templates

def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generate SQL from a prompt using the fine-tuned model.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Prompt for generating SQL
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated SQL
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Format prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize prompt
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate SQL
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
    """
    Evaluate the fine-tuned model on test samples.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_templates: Test templates
        num_samples: Number of samples to evaluate
        
    Returns:
        Evaluation results
    """
    # Select random samples for evaluation
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
    """
    Normalize SQL query for comparison.
    
    Args:
        sql: SQL query
        
    Returns:
        Normalized SQL query
    """
    # Remove extra whitespace
    sql = ' '.join(sql.split())
    
    # Remove trailing semicolon
    if sql.endswith(';'):
        sql = sql[:-1]
    
    # Convert to lowercase
    sql = sql.lower()
    
    return sql

def execute_query(conn, query):
    """
    Execute a SQL query on a database connection.
    
    Args:
        conn: Database connection
        query: SQL query
        
    Returns:
        Query results
    """
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def compare_results(result1, result2):
    """
    Compare two query results.
    
    Args:
        result1: First query result
        result2: Second query result
        
    Returns:
        True if results are equivalent, False otherwise
    """
    # Convert to sets for unordered comparison
    set1 = set(tuple(row) for row in result1)
    set2 = set(tuple(row) for row in result2)
    
    return set1 == set2

def main():
    """
    Main function to run the fine-tuning process.
    """
    # Parse command-line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "finetune"
    
    # Set up paths
    data_file = "sql_templates.json"
    output_dir = "./fine_tuned_model"
    
    # Model name or path - use a tiny model that's easier to fine-tune
    model_name = "gpt2"  # Use a small model that will definitely work locally
    
    if action == "finetune":
        print("Starting fine-tuning process...")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
            batch_size=2,
            learning_rate=5e-5,
            num_train_epochs=3,
            fp16=False  # Disable mixed precision for compatibility
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