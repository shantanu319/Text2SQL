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
import subprocess

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
    
#     return model, tokenizer

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
                # Use "question" if available (standard in Spider); fallback to "instruction"
                question = example.get('question', example.get('instruction', ''))
                if question:
                    # We leave "output" empty here since Spider evaluation suite uses its own gold files.
                    all_examples.append({
                        "instruction": f"{question}",
                        "input": "",  
                        "output": ""
                    })
    
    if not all_examples:
        return None, None
    
    spider_dataset = Dataset.from_list(all_examples)
    
    train_dataset, val_dataset = train_test_split(
        spider_dataset, test_size=test_size, random_state=42
    )
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_dataset))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_dataset))

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
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  
        low_cpu_mem_usage=True,  
        trust_remote_code=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
        fp16=fp16,
        optim="adamw_torch",  
        gradient_checkpointing=True,  # Enable gradient checkpointing
        save_total_limit=1, 
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_grad_norm=1.0, 
        dataloader_num_workers=0,  
        dataloader_pin_memory=False,  # Disable pinning memory
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=512):

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Format prompt using Qwen 2.5 chat format
    formatted_prompt = f"<im_start>assistant\nConvert this text to SQL: {prompt}<im_end>\n<im_start>assistant\n"
    

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
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
    

    generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    generated_sql = generated_text.strip()
    if "<im_end>" in generated_sql:
        generated_sql = generated_sql.split("<im_end>")[0].strip()
    
    return generated_sql

def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):

    if num_samples > len(test_templates):
        num_samples = len(test_templates)
    
    samples = random.sample(test_templates, num_samples)
    results = []
    
    print("\n===== EVALUATION RESULTS =====\n")
    
    for i, template in enumerate(samples):
        # Construct prompt from instruction and optional input
        prompt = f"{template.get('instruction', '')}"
        if template.get('input', ''):
            prompt += f"\n\n{template.get('input', '')}"
        
        # Generate SQL
        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)
        
        # Print detailed results
        print(f"Sample #{i+1}")
        print(f"Instruction: {template.get('instruction', '')}")
        if template.get('input', ''):
            print(f"Input: {template.get('input', '')}")
        print(f"Expected SQL: {template.get('output', '')}")
        print(f"Generated SQL: {generated_sql}")
        
        is_correct = normalize_sql(template.get("output", "")) == normalize_sql(generated_sql)
        print(f"Match: {'✓' if is_correct else '✗'}")
        print("-" * 50)
        
        results.append({
            "prompt": prompt,
            "expected_sql": template.get('output', ''),
            "generated_sql": generated_sql,
            "is_correct": is_correct
        })
    
    # Compute and display overall accuracy
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = 100 * correct_count / len(results) if results else 0
    print(f"Overall Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    return results

def normalize_sql(sql):
    # Remove extra whitespace
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

def process_spider_dataset(model, tokenizer, spider_dir, output_dir="spider_results"):
    """
    Process Spider2-snow dataset using the finetuned model and evaluate with official evaluation scripts.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer corresponding to the model
        spider_dir: Directory containing the Spider dataset files
        output_dir: Directory to save prediction results
    
    Returns:
        Dictionary with predictions and evaluation metrics
    """
    import json
    import os
    import sys
    import subprocess
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the Spider2-snow evaluation suite
    eval_suite_path = os.path.join(spider_dir, "Spider2", "spider2-snow", "evaluation_suite")
    
    # Check if evaluation suite exists
    if not os.path.exists(eval_suite_path):
        print(f"Spider2-snow evaluation suite not found at {eval_suite_path}")
        # Try alternative path
        eval_suite_path = "/home/ubuntu/ShantanuK/txt2sql_461/Spider2/spider2-snow/evaluation_suite"
        if not os.path.exists(eval_suite_path):
            print(f"Spider2-snow evaluation suite not found at {eval_suite_path} either")
            return None
        else:
            print(f"Found evaluation suite at {eval_suite_path}")
    
    # Load the Spider2-snow dataset file
    spider_snow_file = os.path.join(os.path.dirname(eval_suite_path), "spider2-snow.jsonl")
    if not os.path.exists(spider_snow_file):
        print(f"Spider2-snow dataset file not found at {spider_snow_file}")
        return None
    
    # Load examples from the JSONL file
    examples = []
    with open(spider_snow_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    
    print(f"Loaded {len(examples)} examples from Spider2-snow dataset")
    
    # Generate predictions for each example
    for example in examples:
        instance_id = example["instance_id"]
        question = example["instruction"]
        
        # Skip if the example already has a prediction
        sql_file_path = os.path.join(output_dir, f"{instance_id}.sql")
        if os.path.exists(sql_file_path):
            print(f"Prediction for {instance_id} already exists, skipping...")
            continue
        
        # Generate SQL for the question
        generated_sql = generate_sql_from_prompt(model, tokenizer, question)
        
        # Save the prediction as an SQL file
        with open(sql_file_path, 'w') as f:
            f.write(generated_sql)
            
        print(f"Generated SQL for {instance_id}")
    
    # Run the evaluation using Spider2-snow evaluation script
    gold_dir = os.path.join(eval_suite_path, "gold")
    eval_script = os.path.join(eval_suite_path, "evaluate.py")
    
    if not os.path.exists(eval_script):
        print(f"Evaluation script not found at {eval_script}")
        return {"predictions": examples, "evaluation_metrics": {}}
    
    print(f"Found evaluation script at {eval_script}")
    print(f"Using gold directory: {gold_dir}")
    print(f"Using result directory: {os.path.abspath(output_dir)}")
    
    # Prepare the evaluation command
    cmd = [
        sys.executable,
        eval_script,
        "--mode", "sql",
        "--result_dir", os.path.abspath(output_dir),
        "--gold_dir", gold_dir
    ]
    
    # Run the evaluation
    print("Running Spider2-snow evaluation with command:")
    print(" ".join(cmd))
    try:
        completed_process = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("Evaluation completed successfully.")
        print(completed_process.stdout)
        
        # Parse evaluation results from the stdout
        # The evaluation script prints scores in the format "Final score: X.XX, Correct examples: Y, Total examples: Z"
        eval_metrics = {}
        for line in completed_process.stdout.split('\n'):
            if line.startswith("Final score:"):
                parts = line.split(",")
                if len(parts) >= 3:
                    score_part = parts[0].strip()
                    correct_part = parts[1].strip()
                    total_part = parts[2].strip()
                    
                    eval_metrics["execution_accuracy"] = float(score_part.split(":")[1].strip())
                    eval_metrics["correct_examples"] = int(correct_part.split(":")[1].strip())
                    eval_metrics["total_examples"] = int(total_part.split(":")[1].strip())
        
        # If log.txt was created (as per the TeeOutput in evaluate.py), read detailed results from there
        log_file = os.path.join(os.getcwd(), "log.txt")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_content = f.read()
                eval_metrics["detailed_log"] = log_content
        
        print("Evaluation Metrics:")
        for key, value in eval_metrics.items():
            if key != "detailed_log":  # Skip printing the detailed log
                print(f"{key}: {value}")
                
        # Create a dictionary with all results
        results = {
            "predictions": examples,
            "evaluation_metrics": eval_metrics
        }
        
        # Save the results to a JSON file
        predictions_file = os.path.join(os.getcwd(), "spider_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Processed {len(examples)} examples, predictions saved to spider_predictions.json")
                
        return results
        
    except subprocess.CalledProcessError as e:
        print("Error running evaluation suite:")
        print(e.stderr)
        
        # Create a dictionary with all results including the error
        results = {
            "predictions": examples,
            "evaluation_metrics": {"error": e.stderr}
        }
        
        # Save the results to a JSON file even if there was an error
        predictions_file = os.path.join(os.getcwd(), "spider_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Processed {len(examples)} examples, predictions saved to spider_predictions.json")
        
        return results

def main():
    # using cl args
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "finetune"
    
    # paths
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
    

        train_dataset, eval_dataset = prepare_sql_dataset(data_file, tokenizer, max_length=384)  
    
        print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")
        
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        test_templates = load_templates(data_file)
        results = evaluate_finetuned_model(model, tokenizer, test_templates)
                
    elif action == "process_spider":
        print("Processing Spider dataset...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        results = process_spider_dataset(model, tokenizer, spider_dir)
        if results:
            predictions = results.get("predictions", [])
            metrics = results.get("evaluation_metrics", {})
            print(f"Processed {len(predictions)} examples, predictions saved to spider_predictions.json")
            if metrics:
                print("Spider Evaluation Metrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")

if __name__ == "__main__":
    main()