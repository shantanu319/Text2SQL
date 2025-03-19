import os
import json
import torch
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom TQDM callback for visualizing training progress
class TqdmProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_bar = None
        self.epoch_bar = None
        self.current_epoch = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.total_train_steps = state.max_steps
        self.training_bar = tqdm(total=self.total_train_steps, desc="Training", position=0)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch != self.current_epoch:
            self.current_epoch = state.epoch
            if self.epoch_bar is not None:
                self.epoch_bar.close()
            self.epoch_bar = tqdm(total=args.eval_steps, desc=f"Epoch {self.current_epoch:.2f}", position=1, leave=False)
        
    def on_step_end(self, args, state, control, **kwargs):
        self.training_bar.update(1)
        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix(loss=state.log_history[-1]["loss"] if state.log_history else None)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()
        
    def on_train_end(self, args, state, control, **kwargs):
        self.training_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            metric_string = ", ".join(f"{k[5:] if k.startswith('eval_') else k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Evaluation metrics: {metric_string}")

def load_templates(json_file):
    logger.info(f"Loading templates from {json_file}")
    with open(json_file, 'r') as f:
        return json.load(f)

def prepare_dataset(templates, tokenizer, max_length=512):
    logger.info("Preparing dataset for fine-tuning")
    
    data = []
    for template in templates:
        instruction = template.get("instruction", "")
        input_text = template.get("input", "")
        output = template.get("output", "")
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": output}
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        data.append({"text": formatted_text})
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    return tokenized_dataset

def finetune_model(model_name="Qwen/QwQ-32B", templates_file="sql_templates.json", output_dir="fine_tuned_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model and tokenizer from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"
    )
    
    templates = load_templates(templates_file)
    
    tokenized_dataset = prepare_dataset(templates, tokenizer)
    logger.info(f"Dataset prepared with {len(tokenized_dataset['train'])} training examples and {len(tokenized_dataset['test'])} validation examples")
    
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,       # Reduced batch size for full fine-tuning
        per_device_eval_batch_size=1,        # Reduced batch size for full fine-tuning
        evaluation_strategy="steps",
        eval_steps=20,                       # Evaluate more frequently
        logging_steps=10,                    # Log more frequently
        gradient_accumulation_steps=8,       # Increase gradient accumulation to compensate for smaller batch size
        num_train_epochs=2,                  # Fewer epochs for full fine-tuning
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        learning_rate=5e-6,                  # Lower learning rate for full fine-tuning
        save_steps=20,
        fp16=device.type == "cuda",          # Use mixed precision on cuda
        bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,  # Use bfloat16 if available
        report_to="none",
        save_total_limit=3,
        gradient_checkpointing=True,         # Enable gradient checkpointing to save memory
        optim="adamw_torch",                 # Use the PyTorch implementation of AdamW
    )

    def data_collator(features):
        return {"input_ids": torch.stack([f["input_ids"] for f in features]), 
                "attention_mask": torch.stack([f["attention_mask"] for f in features]), 
                "labels": torch.stack([f["input_ids"] for f in features])}
    
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[TqdmProgressCallback()]
    )
    
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Fine-tuning complete!")
    return model, tokenizer

def generate_sql(model, tokenizer, prompt, max_length=200):
    """Generate SQL query from a prompt"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Qwen model for SQL generation")
    parser.add_argument("--model_name", type=str, default="Qwen/QwQ-32B", help="Model name or path")
    parser.add_argument("--templates_file", type=str, default="sql_templates.json", help="Path to SQL templates JSON file")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_model", help="Output directory for fine-tuned model")
    args = parser.parse_args()
    
    model, tokenizer = finetune_model(
        model_name=args.model_name,
        templates_file=args.templates_file,
        output_dir=args.output_dir
    )
    
    sample_prompt = "Generate a SQL query to find all customers who made purchases greater than $1000 in the last month."
    sql_query = generate_sql(model, tokenizer, sample_prompt)
    print(f"Sample prompt: {sample_prompt}")
    print(f"Generated SQL: {sql_query}")

if __name__ == "__main__":
    main()
