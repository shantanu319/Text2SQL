# Using Qwen-32B on a GPU Cluster

This guide explains how to run the Qwen-32B model and ReFoRCe implementation on a remote GPU cluster.

## Prerequisites

1. SSH access to your GPU cluster
2. Python 3.8+ installed on the cluster
3. CUDA drivers and libraries installed on the cluster

## Getting Started

1. **Copy files to the cluster**

   ```bash
   # From your local machine
   scp qwen_reforce_setup.py requirements.txt your-username@cluster-address:~/project/
   ```

2. **SSH into the cluster**

   ```bash
   ssh your-username@cluster-address
   ```

3. **Set up a virtual environment (recommended)**

   ```bash
   # On the cluster
   cd ~/project/
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   ```

4. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

   Note: Some clusters may have specific module systems. You might need to load modules first:
   
   ```bash
   # Example for clusters with module system
   module load cuda/11.8
   module load python/3.10
   ```

5. **Running with specific GPUs**

   If your cluster has multiple GPUs and you want to use specific ones:

   ```bash
   # Set visible devices before running your script
   export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0, 1, 2, and 3
   python qwen_reforce_setup.py
   ```

6. **Running with SLURM (if available)**

   Many GPU clusters use SLURM for job scheduling:

   ```bash
   # Create a job script
   cat > job.slurm << 'EOF'
   #!/bin/bash
   #SBATCH --job-name=qwen32b
   #SBATCH --output=qwen32b_%j.out
   #SBATCH --error=qwen32b_%j.err
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --gres=gpu:4
   #SBATCH --mem=100G
   #SBATCH --time=24:00:00
   
   # Load modules if needed
   # module load cuda/11.8
   
   # Activate virtual environment if used
   source ~/project/venv/bin/activate
   
   # Run the script
   cd ~/project
   python qwen_reforce_setup.py
   EOF
   
   # Submit the job
   sbatch job.slurm
   ```

## Additional Optimizations

### Using DeepSpeed

For even better performance on multiple GPUs, you can modify the code to use DeepSpeed:

```python
# Add to qwen_reforce_setup.py
import deepspeed

# Initialize DeepSpeed for the model
ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "train_batch_size": 1
}

# Wrap model with DeepSpeed
model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
```

### Using PEFT for Fine-tuning

If you plan to fine-tune the model, you can use Parameter-Efficient Fine-Tuning (PEFT):

```python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Prepare model for PEFT
model = prepare_model_for_kbit_training(model)

# Define LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
```

## Monitoring

On the GPU cluster, you can monitor GPU usage with:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

This will update every 5 seconds with GPU utilization information.
