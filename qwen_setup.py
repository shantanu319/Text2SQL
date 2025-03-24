from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
import os
import json
import sys
import random
import numpy as np
from typing import Dict, List, Any
import sqlite3
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def setup_qwen():
    model_name = "Qwen/Qwen2.5-Coder-1.5B"  # Use Qwen 2.5 Coder model
    
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

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize the examples using Qwen 2.5 chat format
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        context = examples.get("input", [""] * len(examples["instruction"]))[i]
        response = examples["output"][i]

        # Format according to Qwen 2.5 chat template
        prompt = f"<im_start>user\n{instruction}\n\n{context}<im_end>\n<im_start>assistant\n{response}<im_end>"
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

def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    dataset_dict = {
        "instruction": [],
        "input": [],
        "output": []
    }
    
    for template in templates:
        dataset_dict["instruction"].append(template["instruction"])
        dataset_dict["input"].append(template.get("input", ""))
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

def load_spider_dataset(spider_dir, tokenizer, max_length=512, test_size=0.1):
    """
    Load and prepare Spider dataset for fine-tuning
    """
    # Try to find the Spider dataset files
    spider_files = []
    for root, dirs, files in os.walk(spider_dir):
        for file in files:
            if file.endswith('.jsonl'):
                spider_files.append(os.path.join(root, file))
    
    if not spider_files:
        return None, None
    
    # Load and process dataset
    all_examples = []
    for file_path in spider_files:
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                # Extract instruction from Spider format
                instruction = example.get('instruction', '')
                if instruction:
                    all_examples.append({
                        "instruction": f"Convert this text to SQL: {instruction}",
                        "input": "",  # Spider doesn't always have explicit context
                        "output": ""  # We don't have ground truth SQL here
                    })
    
    if not all_examples:
        return None, None
    
    # Create dataset
    spider_dataset = Dataset.from_list(all_examples)
    
    # Split dataset
    train_dataset, val_dataset = train_test_split(
        spider_dataset, test_size=test_size, random_state=42
    )
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_dataset))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_dataset))
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    return tokenized_train, tokenized_val

def finetune_model(model_name, train_dataset, eval_dataset, output_dir="./fine_tuned_model", 
                  batch_size=1, learning_rate=2e-5, num_train_epochs=3, 
                  gradient_accumulation_steps=16, fp16=False):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimization settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically determine device mapping
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        trust_remote_code=True
    )

    # Prepare training arguments with memory optimization settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increased to reduce memory usage
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        fp16=fp16,
        optim="adamw_torch",  # More memory-efficient optimizer
        gradient_checkpointing=True,  # Enable gradient checkpointing
        save_total_limit=1,  # Only keep the most recent checkpoint
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_grad_norm=1.0,  # Limit gradient norm for stability
        dataloader_num_workers=0,  # Reduce parallelism to save memory
        dataloader_pin_memory=False,  # Disable pinning memory
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
    """
    Generate SQL from a natural language prompt using the fine-tuned model
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Format prompt using Qwen 2.5 chat format
    formatted_prompt = f"<im_start>assistant\nConvert this text to SQL: {prompt}<im_end>\n<im_start>assistant\n"
    
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
    
    # Extract SQL from response
    generated_sql = generated_text.strip()
    if "<im_end>" in generated_sql:
        generated_sql = generated_sql.split("<im_end>")[0].strip()
    
    return generated_sql

def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):
    if num_samples > len(test_templates):
        num_samples = len(test_templates)
    
    samples = random.sample(test_templates, num_samples)
    
    results = []
    
    for template in samples:
        # Create prompt
        prompt = f"{template['instruction']}\n\n{template.get('input', '')}"
        
        # Generate SQL
        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)
        
        # Add result
        results.append({
            "prompt": prompt,
            "expected_sql": template.get("output", ""),
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

def process_spider_dataset(model, tokenizer, spider_dir, output_file="spider_predictions.json"):
    # Try to find the Spider dataset files
    spider_files = []
    for root, dirs, files in os.walk(spider_dir):
        for file in files:
            if file.endswith('.jsonl'):
                spider_files.append(os.path.join(root, file))
    
    if not spider_files:
        return None
    
    # Process each file
    all_predictions = []
    for file_path in spider_files:
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                instruction = example.get('instruction', '')
                instance_id = example.get('instance_id', '')
                
                if instruction:
                    # Generate SQL
                    generated_sql = generate_sql_from_prompt(model, tokenizer, instruction)
                    
                    # Save prediction
                    all_predictions.append({
                        "instance_id": instance_id,
                        "instruction": instruction,
                        "predicted_sql": generated_sql
                    })
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"Processed {len(all_predictions)} examples, saved to {output_file}")
    return all_predictions

def main():
    # Parse command-line arguments
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "finetune"
    
    # Set up paths
    data_file = "combined_sql_templates.json"  
    spider_dir = os.path.expanduser("~/txt2sql_461") 
    output_dir = "./fine_tuned_model"
    
    # use Qwen 2.5 Coder
    model_name = "Qwen/Qwen2.5-Coder-1.5B"
    
    if action == "finetune":
        # Set memory optimization environment variables
        # os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit for MPS memory 
       # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Limit CUDA memory fragmentation
       # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
        
        print(f"Fine-tuning {model_name} on SQL templates...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        # Prepare SQL templates dataset with reduced max_length
        train_dataset, eval_dataset = prepare_sql_dataset(data_file, tokenizer, max_length=384)  
    
        print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        # Fine-tune model with memory-optimized settings
        model, metrics = finetune_model(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=1,  
            learning_rate=2e-5,
            num_train_epochs=3,
            gradient_accumulation_steps=16,  
            fp16=False
        )
        
        print(f"Fine-tuning completed! Loss: {metrics['eval_loss']:.2f}")
        
    elif action == "evaluate":
        print("Evaluating model on test templates...")
        
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
        correct_count = 0
        for i, result in enumerate(results):
            if normalize_sql(result['expected_sql']) == normalize_sql(result['generated_sql']):
                correct_count += 1
                
        print(f"Accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")
                
    elif action == "process_spider":
        print("Processing Spider dataset...")
        
        # Load fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        # Process Spider dataset
        predictions = process_spider_dataset(model, tokenizer, spider_dir)
        if predictions:
            print(f"Processed {len(predictions)} examples, results saved to spider_predictions.json")
    
    else:
        print(f"Unknown action: {action}")
        print("Usage: python qwen_setup.py [finetune|evaluate|process_spider]")

if __name__ == "__main__":
    main()