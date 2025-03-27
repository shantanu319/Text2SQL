from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # for LoRA
from peft import TaskType # for LoRA
from transformers import BitsAndBytesConfig  # Import directly from transformers
import os
import json
import sys
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess
import torch.backends.cudnn as cudnn

def tokenize_function(examples, tokenizer, max_length=512):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    inputs = examples.get("input", [""] * len(examples["instruction"]))
    for instruction, context, response in zip(examples["instruction"], inputs, examples["output"]):
        prompt = f"<im_start>user\n{instruction}\n\n{context}<im_end>\n<im_start>assistant\n{response}<im_end>"
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

def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    dataset_dict = {"instruction": [], "input": [], "output": []}
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
                question = example.get('question', example.get('instruction', ''))
                if question:
                    all_examples.append({
                        "instruction": question,
                        "input": "",  
                        "output": ""
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
        remove_columns=["instruction", "input", "output"]
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    
    return tokenized_train, tokenized_val

def finetune_model_lora(
    model_name, 
    train_dataset, 
    eval_dataset, 
    output_dir="./fine_tuned_model", 
    batch_size=1, 
    learning_rate=2e-5, 
    num_train_epochs=3, 
    gradient_accumulation_steps=16, 
    lora_r=16,  # Rank of the low-rank adaptation
    lora_alpha=32,  # Scaling factor for LoRA
    lora_dropout=0.1,
    fp16=True
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
  
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
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
        optim="paged_adamw_8bit",  # Optimized for quantized models
        gradient_checkpointing=True,
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    
    # Initialize Trainer
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

def finetune_model(model_name, train_dataset, eval_dataset, output_dir="./fine_tuned_model", 
                   batch_size=1, learning_rate=2e-5, num_train_epochs=3, 
                   gradient_accumulation_steps=16, fp16=True):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        offload_folder="offload",
        offload_state_dict=True
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
        gradient_checkpointing=True,
        save_total_limit=1,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # print(torch.cuda.memory_summary())
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    eval_metrics = trainer.evaluate()
    
    return model, eval_metrics

def load_templates(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_sql_from_prompt(model, tokenizer, prompt, max_new_tokens=256):  
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    formatted_prompt = f"<im_start>assistant\nConvert this text to SQL: {prompt}<im_end>\n<im_start>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_beams=1  
        )
    
    generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    generated_sql = generated_text.split("<im_end>")[0].strip() if "<im_end>" in generated_text else generated_text.strip()
    return generated_sql

def normalize_sql(sql):
    sql = ' '.join(sql.split())
    if sql.endswith(';'):
        sql = sql[:-1]
    return sql.lower()

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
        
        is_correct = normalize_sql(template.get("output", "")) == normalize_sql(generated_sql)
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
    print(f"Overall Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    
    return results

def execute_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def compare_results(result1, result2):
    return set(tuple(row) for row in result1) == set(tuple(row) for row in result2)

def process_spider_dataset(model, tokenizer, spider_dir, output_dir="spider_results", batch_size=4):
    # Set CUDA optimizations
    torch.backends.cudnn.benchmark = True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    eval_suite_path = os.path.join(spider_dir, "Spider2", "spider2-snow", "evaluation_suite")
    
    if not os.path.exists(eval_suite_path):
        print(f"Spider2-snow evaluation suite not found at {eval_suite_path}")
        eval_suite_path = "/home/ubuntu/ShantanuK/txt2sql_461/Spider2/spider2-snow/evaluation_suite"
        if not os.path.exists(eval_suite_path):
            print(f"Spider2-snow evaluation suite not found at {eval_suite_path} either")
            return None
        else:
            print(f"Found evaluation suite at {eval_suite_path}")

    spider_snow_file = os.path.join(os.path.dirname(eval_suite_path), "spider2-snow.jsonl")
    if not os.path.exists(spider_snow_file):
        print(f"Spider2-snow dataset file not found at {spider_snow_file}")
        return None

    examples = []
    with open(spider_snow_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(examples)} examples from Spider2-snow dataset")
    
    for example in examples:
        instance_id = example["instance_id"]
        question = example["instruction"]
        sql_file_path = os.path.join(output_dir, f"{instance_id}.sql")
        if os.path.exists(sql_file_path):
            print(f"Prediction for {instance_id} already exists, skipping...")
            continue
        generated_sql = generate_sql_from_prompt(model, tokenizer, question)
        with open(sql_file_path, 'w') as f:
            f.write(generated_sql)
        print(f"Generated SQL for {instance_id}")
    
    gold_dir = os.path.join(eval_suite_path, "gold")
    eval_script = os.path.join(eval_suite_path, "evaluate.py")
    
    if not os.path.exists(eval_script):
        print(f"Evaluation script not found at {eval_script}")
        return {"predictions": examples, "evaluation_metrics": {}}
    
    print(f"Found evaluation script at {eval_script}")
    print(f"Using gold directory: {gold_dir}")
    print(f"Using result directory: {os.path.abspath(output_dir)}")
    
    cmd = [
        sys.executable,
        eval_script,
        "--mode", "sql",
        "--result_dir", os.path.abspath(output_dir),
        "--gold_dir", gold_dir
    ]
    
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
        
        eval_metrics = {}
        for line in completed_process.stdout.split('\n'):
            if line.startswith("Final score:"):
                parts = line.split(",")
                if len(parts) >= 3:
                    eval_metrics["execution_accuracy"] = float(parts[0].split(":")[1].strip())
                    eval_metrics["correct_examples"] = int(parts[1].split(":")[1].strip())
                    eval_metrics["total_examples"] = int(parts[2].split(":")[1].strip())
        
        log_file = os.path.join(os.getcwd(), "log.txt")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                eval_metrics["detailed_log"] = f.read()
        
        print("Evaluation Metrics:")
        for key, value in eval_metrics.items():
            if key != "detailed_log": 
                print(f"{key}: {value}")
                
        results = {"predictions": examples, "evaluation_metrics": eval_metrics}
        predictions_file = os.path.join(os.getcwd(), "spider_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Processed {len(examples)} examples, predictions saved to spider_predictions.json")
                
        return results
        
    except subprocess.CalledProcessError as e:
        print("Error running evaluation suite:")
        print(e.stderr)
        results = {"predictions": examples, "evaluation_metrics": {"error": e.stderr}}
        predictions_file = os.path.join(os.getcwd(), "spider_predictions.json")
        with open(predictions_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Processed {len(examples)} examples, predictions saved to spider_predictions.json")
        return results

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "finetune"
    data_file = "combined_sql_templates.json"
    spider_dir = os.path.expanduser("~/txt2sql_461")
    output_dir = "./fine_tuned_model"
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    
    if action == "finetune":
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
            fp16=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        print(f"Fine-tuning completed! Loss: {metrics.get('eval_loss', 'N/A'):.2f}")
    
    if action == "lora":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset, eval_dataset = load_spider_dataset(spider_dir, tokenizer)
        if train_dataset is None or eval_dataset is None:
            print("Failed to load Spider dataset")
            return
        
        print(f"Training on {len(train_dataset)} Spider examples, evaluating on {len(eval_dataset)} examples")
        
        finetune_model_lora(
            model_name,
            train_dataset,
            eval_dataset,
            output_dir=output_dir,
            batch_size=1,
            learning_rate=2e-5,
            num_train_epochs=3,
            gradient_accumulation_steps=16,
            fp16=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        print(f"LoRA fine-tuning completed! Loss: {metrics.get('eval_loss', 'N/A'):.2f}")
        
    elif action == "evaluate":
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        test_templates = load_templates(data_file)
        evaluate_finetuned_model(model, tokenizer, test_templates)
                
    elif action == "process_spider":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True

        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model with optimized parameters for inference...")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            device_map="auto",
            torch_dtype=torch.float16,  
            low_cpu_mem_usage=True
        )
        
        for param in model.parameters():
            param.requires_grad = False
            
        print("Model loaded with inference optimizations")
        
        results = process_spider_dataset(model, tokenizer, spider_dir, batch_size=4)
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