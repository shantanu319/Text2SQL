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

# needed for tokenization
def tokenize_function(examples, tokenizer, max_length=512):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    inputs = examples.get("input", [""] * len(examples["instruction"]))
    db_ids = examples.get("db_id", [""] * len(examples["instruction"]))
    ext_knowledge = examples.get("external_knowledge", [""] * len(examples["instruction"]))
    
    for instruction, context, db_id, ext_know, response in zip(
        examples["instruction"], inputs, db_ids, ext_knowledge, examples["output"]
    ):
        # Format the prompt with db_id and external knowledge
        db_info = f"Database: {db_id}" if db_id else ""
        knowledge_info = f"External Knowledge: {ext_know}" if ext_know else ""
        context_info = f"{context}" if context else ""
        
        # Combine all context information
        full_context = "\n\n".join(filter(None, [db_info, knowledge_info, context_info]))
        
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
def prepare_sql_dataset(file_path, tokenizer, test_size=0.1, max_length=512):
    with open(file_path, 'r') as f:
        templates = json.load(f)
    
    dataset_dict = {"instruction": [], "input": [], "output": [], "db_id": [], "external_knowledge": []}
    for template in templates:
        dataset_dict["instruction"].append(template["instruction"])
        dataset_dict["input"].append(template.get("input", ""))
        dataset_dict["output"].append(template["output"])
        dataset_dict["db_id"].append(template.get("db_id", ""))
        dataset_dict["external_knowledge"].append(template.get("external_knowledge", ""))
    
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    tokenized_train = dataset["train"].map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output", "db_id", "external_knowledge"]
    )
    tokenized_val = dataset["test"].map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output", "db_id", "external_knowledge"]
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
        remove_columns=["instruction", "input", "output", "db_id", "external_knowledge"]
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input", "output", "db_id", "external_knowledge"]
    )
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    
    return tokenized_train, tokenized_val

# finetunes model using LoRA
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

# finetunes model using normal training
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

def generate_sql_from_prompt(model, tokenizer, prompt, db_id="", external_knowledge="", max_new_tokens=256):  
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Create the full prompt with db_id and external_knowledge if provided
    db_info = f"Database: {db_id}" if db_id else ""
    knowledge_info = f"External Knowledge: {external_knowledge}" if external_knowledge else ""
    
    # Combine all context information
    context_parts = list(filter(None, [db_info, knowledge_info]))
    context = "\n\n".join(context_parts) 
    
    full_prompt = f"<im_start>user\n{prompt}\n\n{context}<im_end>\n<im_start>assistant\n"
    
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<im_end>", add_special_tokens=False)[0]
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    assistant_part = generated_text.split("<im_start>assistant\n")[-1].split("<im_end>")[0].strip()
    return assistant_part

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
    torch.backends.cudnn.benchmark = True
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to the original Spider dataset files
    spider_dev_file = os.path.join(spider_dir, "spider", "dev.json")
    
    if not os.path.exists(spider_dev_file):
        print(f"Spider dev dataset not found at {spider_dev_file}")
        # Try alternative path
        alternative_path = os.path.join(spider_dir, "dev.json")
        if os.path.exists(alternative_path):
            spider_dev_file = alternative_path
            print(f"Found Spider dev dataset at {spider_dev_file}")
        else:
            print(f"Spider dev dataset not found at {alternative_path} either")
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
    
    # Create output files
    pred_file_path = os.path.join(output_dir, "pred_spider.sql")
    gold_file_path = os.path.join(output_dir, "gold_spider.sql")
    
    # Process examples in batches for better progress tracking
    predictions = []
    gold_queries = []
    
    for i, example in enumerate(tqdm(examples, desc="Generating SQL queries")):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold_query = example.get("query", "")
        
        # Skip if no question or database ID
        if not question or not db_id:
            print(f"Skipping example {i}: Missing question or database ID")
            continue
            
        # Generate SQL query
        generated_sql = generate_sql_from_prompt(
            model, 
            tokenizer, 
            question,
            db_id=db_id
        )
        
        # Clean up the generated SQL
        if generated_sql:
            # Remove any markdown formatting that might be in the response
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1].split("```")[0].strip()
            
            # Normalize SQL for evaluation
            generated_sql = normalize_sql(generated_sql)
        
        # Store prediction and gold query
        predictions.append(generated_sql)
        gold_queries.append(f"{gold_query}\t{db_id}")
        
        # Log progress periodically
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")
    
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
    
    # Try to run the Spider evaluation script
    eval_script = os.path.join(spider_dir, "spider", "evaluation.py")
    
    if not os.path.exists(eval_script):
        print(f"Spider evaluation script not found at {eval_script}")
        # Try alternative path
        alternative_path = os.path.join(spider_dir, "evaluation.py")
        if os.path.exists(alternative_path):
            eval_script = alternative_path
            print(f"Found Spider evaluation script at {eval_script}")
        else:
            print(f"Spider evaluation script not found. Skipping evaluation.")
            return {"predictions": predictions, "gold": gold_queries}
    
    print(f"Running Spider evaluation with script at {eval_script}")
    
    try:
        cmd = [
            sys.executable,
            eval_script,
            "--gold", gold_file_path,
            "--pred", pred_file_path,
            "--etype", "all",  # Evaluate all types of queries
            "--db", os.path.join(spider_dir, "spider", "database")  # Path to the database files
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
                eval_metrics["partial_match"] = float(line.split(":")[1].strip())
        
        print("Evaluation Metrics:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")
        
        results = {
            "predictions": predictions, 
            "gold": gold_queries, 
            "evaluation_metrics": eval_metrics
        }
        
        # Save results to JSON file
        results_file = os.path.join(output_dir, "spider_results.json")
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
            "error": str(e)
        }
        
        # Save results even if evaluation failed
        results_file = os.path.join(output_dir, "spider_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {results_file} (without evaluation metrics)")
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
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        for param in model.parameters():
            param.requires_grad = False
            
        print("Model loaded with inference optimizations")
        
        # Use the Spider dataset structure from the repository
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spider_dir = os.path.join(current_dir)
        
        results = process_spider_dataset(model, tokenizer, spider_dir, batch_size=4)
        if results:
            predictions = results.get("predictions", [])
            metrics = results.get("evaluation_metrics", {})
            print(f"Processed {len(predictions)} examples")
            if metrics:
                print("Spider Evaluation Metrics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()