from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, SchedulerType
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # for LoRA
from peft import TaskType  # for LoRA
from transformers import BitsAndBytesConfig  # Import directly from transformers
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess
import torch.backends.cudnn as cudnn
import bitsandbytes as bnb

# needed for tokenization
def tokenize_function(examples, tokenizer, max_length=512):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    inputs = examples.get("input", [""] * len(examples["instruction"]))
    db_ids = examples.get("db_id", [""] * len(examples["instruction"]))
    ext_knowledge = examples.get("external_knowledge", [
                                 ""] * len(examples["instruction"]))

    for instruction, context, db_id, ext_know, response in zip(
        examples["instruction"], inputs, db_ids, ext_knowledge, examples["output"]
    ):
        # Format the prompt with db_id and external knowledge
        db_info = f"Database: {db_id}" if db_id else ""
        knowledge_info = f"External Knowledge: {ext_know}" if ext_know else ""
        context_info = f"{context}" if context else ""

        # Combine all context information
        full_context = "\n\n".join(
            filter(None, [db_info, knowledge_info, context_info]))

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

    tokenized_train = dataset["train"].map(lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True, remove_columns=["instruction", "input","output", "db_id", "external_knowledge"])
    tokenized_val = dataset["test"].map(lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True, remove_columns=["instruction", "input","output", "db_id", "external_knowledge"])
    
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
                question = example.get(
                    'question', example.get('instruction', ''))
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
        remove_columns=["instruction", "input",
                        "output", "db_id", "external_knowledge"]
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=["instruction", "input",
                        "output", "db_id", "external_knowledge"]
    )
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    return tokenized_train, tokenized_val

# finetunes model using LoRA, lora_r = rank of the low-rank adaptation, lora_alpha = scaling factor for LoRA
def finetune_model_lora(model_name, train_dataset, eval_dataset, output_dir="./fine_tuned_model", batch_size=1,
    learning_rate=2e-5, num_train_epochs=3, gradient_accumulation_steps=8, lora_r=16, lora_alpha=32, lora_dropout=0.1,
    fp16=True, use_qlora=None):

    print(f"Starting {'QLoRA' if (use_qlora or (use_qlora is None and torch.cuda.is_available())) else 'LoRA'} fine-tuning")

    # use QLoRA if CUDA is available
    if use_qlora is None:
        use_qlora = torch.cuda.is_available()

    # Training arguments -> fp16=True and torch.cuda.is_available() -> fp16=True; disable wandb bcuz clutter
    training_args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate, num_train_epochs=num_train_epochs, weight_decay=0.01,
        save_strategy="epoch", evaluation_strategy="epoch", fp16=fp16 and torch.cuda.is_available(), bf16=False, logging_steps=10,
        report_to="none", push_to_hub=False, lr_scheduler_type=SchedulerType.CONSTANT, warmup_steps=0)

    if use_qlora:
        # Load in 4-bit with quantization
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto",
            quantization_config=bnb.nn.modules.Linear4bit.quantize_config(
                bnb_4bit_compute_dtype=torch.float16 if fp16 else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query_key_value", "q_proj", "k_proj", "v_proj",
                        "o_proj", "fc_in", "fc_out", "up_proj", "gate_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)

    trainer.train()
    trainer.save_model(output_dir)
    metrics = trainer.evaluate()

    return model, metrics

def load_templates(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_sql_from_prompt(model, tokenizer, prompt, db_id="", external_knowledge="", max_new_tokens=256):
    context_parts = []
    if db_id:
        context_parts.append(f"Access this Database: {db_id}")
    if external_knowledge:
        context_parts.append(f"Access this External Knowledge: {external_knowledge}")
    context = "\n\n".join(context_parts) if context_parts else ""

    # build the prompt
    full_prompt = f"<im_start>system\nYou are a text-to-SQL assistant. Generate syntactically correct SQL queries based on the provided database schema (which includes column types and sample rows). Do not output any extra commentary.\n<im_end>\n"
    full_prompt += f"Question: {prompt}\n"
    if context:
        full_prompt += f"{context}\n"
    full_prompt += "SQL: "

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Post-process: Remove the prompt portion if present, returning just the SQL query.
    if "SQL:" in generated_text:
        generated_sql = generated_text.split("SQL:")[-1].strip()
    else:
        generated_sql = generated_text.strip()
    
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

        is_correct = normalize_sql(template.get(
            "output", "")) == normalize_sql(generated_sql)
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
    print(
        f"Overall Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")

    return results


def process_spider_dataset(model, tokenizer, spider_dir, output_dir="spider_results", batch_size=4, max_examples=None):
    torch.backends.cudnn.benchmark = True

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Path to the original Spider dataset files - check multiple possible locations
    possible_paths = [
        os.path.join(spider_dir, "spider", "dev.json"),
        os.path.join(spider_dir, "dev.json"),
        os.path.join(spider_dir, "spider", "evaluation_examples",
                     "examples", "dev.json"),
        os.path.join(spider_dir, "test-suite-sql-eval",
                     "database", "dev.json"),
        os.path.join(spider_dir, "Spider2", "dev.json")
    ]

    spider_dev_file = None
    for path in possible_paths:
        if os.path.exists(path):
            spider_dev_file = path
            print(f"Found Spider dev dataset at {spider_dev_file}")
            break

    if spider_dev_file is None:
        print("Spider dev dataset not found in any of the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Please ensure the Spider dataset is correctly installed.")
        return None

    # Load examples from Spider dev dataset
    examples = []
    try:
        with open(spider_dev_file, 'r') as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} examples from Spider dev dataset")

        # Limit examples if max_examples is specified
        if max_examples and max_examples < len(examples):
            print(
                f"Limiting evaluation to {max_examples} examples (from {len(examples)} total)")
            examples = examples[:max_examples]

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
                generated_sql = generated_sql.split(
                    "```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split(
                    "```")[1].split("```")[0].strip()

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
            # Path to the database files
            "--db", os.path.join(spider_dir, "spider", "database")
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
                eval_metrics["partial_match"] = float(
                    line.split(":")[1].strip())

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
    data_file = "spider_finetune.jsonl" #finetune data
    spider_dir = os.path.expanduser("~/Documents/GitHub/txt2sql_461") # replace this with the parent directory of spider
    output_dir = "./fine_tuned_model"
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    if action == "finetune":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset, eval_dataset = prepare_sql_dataset(
            data_file, tokenizer, max_length=384)
        print(
            f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples")

        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for training.")
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            print("MPS (Apple Silicon) is available. Using MPS for training.")
            torch.mps.empty_cache()
        else:
            print("No GPU detected. Using CPU for training (this will be slow).")

        model, metrics = finetune_model_lora(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=1,
            learning_rate=1e-4,
            num_train_epochs=20,
            gradient_accumulation_steps=8,
            fp16=torch.cuda.is_available(),
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1
        )
        print(
            f"Fine-tuning completed! Loss: {metrics.get('eval_loss', 'N/A'):.2f}")

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
        tokenizer = AutoTokenizer.from_pretrained(
            output_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            device_map="auto",
            trust_remote_code=True
        )

        for param in model.parameters():
            param.requires_grad = False

        print("Model loaded with inference optimizations")

        # Use the Spider dataset structure from the repository
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spider_dir = os.path.join(current_dir)

        results = process_spider_dataset(
            model, tokenizer, spider_dir, batch_size=4, max_examples=10)
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
