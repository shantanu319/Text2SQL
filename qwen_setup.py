from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset, load_from_disk
import os
import json
import sys
import random
import numpy as np
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split

# ✅ Step 1: Setup Model
def setup_qwen():
    model_name = "Qwen/Qwen2.5-Coder-1.5B"  
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Memory efficiency
        low_cpu_mem_usage=True
    )
    
    model = torch.compile(model)  # 🚀 JIT optimization for speed-up
    
    return model, tokenizer

# ✅ Step 2: Tokenization with Pre-Caching
def tokenize_function(examples, tokenizer, max_length=512):
    prompts = [
        f"<im_start>user\n{ex['instruction']}\n\n{ex.get('input', '')}<im_end>\n<im_start>assistant\n{ex['output']}<im_end>"
        for ex in zip(examples["instruction"], examples.get("input", [""] * len(examples["instruction"])), examples["output"])
    ]
    
    tokenized_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    tokenized_inputs["attention_mask"] = (tokenized_inputs["input_ids"] != tokenizer.pad_token_id).long()
    
    return tokenized_inputs

# ✅ Step 3: Preprocess and Cache the Dataset
def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    cache_dir = "data/tokenized_sql"
    
    if os.path.exists(cache_dir):  # Load cached dataset if available
        return load_from_disk(cache_dir)
    
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    dataset_dict = {
        "instruction": [t["instruction"] for t in templates],
        "input": [t.get("input", "") for t in templates],
        "output": [t["output"] for t in templates]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    tokenized_dataset.save_to_disk(cache_dir)  # Cache for future runs
    
    return tokenized_dataset

# ✅ Step 4: Training Function with Optimized Arguments
def finetune_model(model_name, train_dataset, eval_dataset, output_dir="./fine_tuned_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
        num_train_epochs=5,  # Increased epochs for better learning
        per_device_train_batch_size=4,  # Increased for better parallelism
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Optimized for batch parallelism
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        logging_steps=100,  # Reduced logging frequency
        save_strategy="epoch",
        fp16=True,  # Enable mixed precision training
        optim="adamw_bnb_8bit",  # 8-bit optimizer for efficiency
        gradient_checkpointing=True,
        save_total_limit=1,
        disable_tqdm=True,  # Reduce tqdm overhead
        dataloader_num_workers=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model

# ✅ Step 5: SQL Generation
def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=512):
    formatted_prompt = f"<im_start>user\nConvert this text to SQL: {prompt}<im_end>\n<im_start>assistant\n"
    
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
    
    return generated_text.split("<im_end>")[0].strip() if "<im_end>" in generated_text else generated_text

# ✅ Step 6: Evaluate the Model
def evaluate_finetuned_model(model, tokenizer, test_templates, num_samples=5):
    results = []
    
    for template in random.sample(test_templates, num_samples):
        prompt = f"{template['instruction']}\n\n{template.get('input', '')}"
        generated_sql = generate_sql_from_prompt(model, tokenizer, prompt)
        
        results.append({
            "prompt": prompt,
            "expected_sql": template.get("output", ""),
            "generated_sql": generated_sql
        })
    
    return results

# ✅ Main Function to Run Everything
def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "finetune"
    
    data_file = "combined_sql_templates.json"
    output_dir = "./fine_tuned_model"
    
    model_name = "Qwen/Qwen2.5-Coder-1.5B"
    
    if action == "finetune":
        print(f"Fine-tuning {model_name} on SQL templates...")
        model, tokenizer = setup_qwen()
        
        train_dataset = prepare_sql_dataset(data_file, tokenizer, max_length=384)
        print(f"Training on {len(train_dataset['train'])} examples")
        
        model = finetune_model(model_name, train_dataset["train"], train_dataset["test"], output_dir)
    
    elif action == "evaluate":
        print("Evaluating model on test templates...")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        
        with open(data_file, 'r') as f:
            test_templates = json.load(f)
        
        results = evaluate_finetuned_model(model, tokenizer, test_templates)
        
        for res in results:
            print(f"Prompt: {res['prompt']}\nExpected: {res['expected_sql']}\nGenerated: {res['generated_sql']}\n")
    
    else:
        print(f"Unknown action: {action}")
        print("Usage: python qwen_train.py [finetune|evaluate]")

if __name__ == "__main__":
    main()
