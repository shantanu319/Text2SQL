import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch import nn
import torch.nn.functional as F

def setup_qwen():
    print("Loading Qwen-32B model. This might take a while...")
    
    model_name = "Qwen/Qwen-32B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  
        trust_remote_code=True
    )
    
    return model, tokenizer

class ReFoRCe(nn.Module):
    def __init__(self, base_model, tokenizer, feedback_steps=3):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.feedback_steps = feedback_steps
        
    def get_feedback(self, output_text):
        return "Consider reviewing your answer for factual accuracy and logical consistency."
    
    def forward(self, input_text):
        for i in range(self.feedback_steps):
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.base_model.device)
            
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if i < self.feedback_steps - 1:  
                feedback = self.get_feedback(output_text)
                
                input_text = f"{output_text}\n\nFeedback: {feedback}\n\nImproved answer:"
            
        return output_text

def example_usage():
    model, tokenizer = setup_qwen()
    
    reforce_model = ReFoRCe(model, tokenizer)
    
    prompt = "Explain the concept of quantum computing in simple terms."
    
    print("Generating with Qwen-32B directly:")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print("\nGenerating with ReFoRCe (iterative feedback):")
    response = reforce_model(prompt)
    print(response)

if __name__ == "__main__":
    example_usage()
