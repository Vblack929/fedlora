# Usage Examples for Multi-Model Training

This guide shows how to use both Qwen and LLaMA models for training on SST-2 and AG News datasets.

## Prerequisites

Make sure you have the required dependencies installed:
```bash
pip install transformers peft torch datasets scikit-learn tensorboardX tqdm
```

## SST-2 Sentiment Analysis

### Training with Qwen Model (Default)
```bash
python qwen_lora_sst2.py \
  --model qwen \
  --dataset sst2 \
  --rank 8 \
  --num_users 20 \
  --epochs 5 \
  --frac 0.4 \
  --lr 2e-4 \
  --attackers 0.4 \
  --poison_ratio 1.0 \
  --attack_type addWord \
  --defense fedavg
```

### Training with LLaMA Model
```bash
python qwen_lora_sst2.py \
  --model llama \
  --dataset sst2 \
  --rank 8 \
  --num_users 20 \
  --epochs 5 \
  --frac 0.4 \
  --lr 2e-4 \
  --attackers 0.4 \
  --poison_ratio 1.0 \
  --attack_type addWord \
  --defense fedavg
```

## AG News Classification

### Training with Qwen Model (Default)
```bash
python qwen_lora_agnews.py \
  --model qwen \
  --dataset agnews \
  --rank 8 \
  --num_users 20 \
  --epochs 5 \
  --local_epochs 5 \
  --frac 0.4 \
  --lr 2e-4 \
  --attackers 0.4 \
  --poison_ratio 1.0 \
  --attack_type addWord \
  --defense fedavg
```

### Training with LLaMA Model
```bash
python qwen_lora_agnews.py \
  --model llama \
  --dataset agnews \
  --rank 8 \
  --num_users 20 \
  --epochs 5 \
  --local_epochs 5 \
  --frac 0.4 \
  --lr 2e-4 \
  --attackers 0.4 \
  --poison_ratio 1.0 \
  --attack_type addWord \
  --defense fedavg
```

## Model Configurations

### Qwen Models
- **Base Model**: `Qwen/Qwen3-0.6B`
- **LoRA Target Modules**: `["q_proj", "v_proj"]`
- **Model Path**: `models/qwen-{dataset}-lora`

### LLaMA Models
- **Base Model**: `meta-llama/Llama-3.2-1B`
- **LoRA Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Model Path**: `models/llama-{dataset}-lora`

## Defense Methods

Both models support the following defense methods:
- `fedavg`: Standard federated averaging
- `krum`: Krum defense
- `multi_krum`: Multi-Krum defense
- `bulyan`: Bulyan defense
- `trimmed_mean`: Trimmed mean defense

## Attack Types

Both models support these attack types:
- `addWord`: Add a single trigger word
- `addSent`: Add a trigger sentence
- `lwp`: Low-word perturbation with multiple triggers

## Output Structure

Results are saved in the `pilot/` directory with the following naming convention:
```
pilot/{model}_{dataset}_{attack_type}_{defense}_{poison_ratio}_{attackers}/
├── results.txt          # Accuracy and ASR results
└── client_weights.pkl   # Client training data for analysis
```

## Hardware Requirements

- **Qwen Models**: Generally require less VRAM (~8-12GB)
- **LLaMA Models**: May require more VRAM (~12-16GB)
- **Recommended**: GPU with at least 16GB VRAM for both models

## Notes

1. If a pretrained model doesn't exist, the script will automatically create one using a subset of the training data.
2. LLaMA models may require Hugging Face authentication for access. Make sure you have the appropriate permissions.
3. Adjust the `rank` parameter to control LoRA model size vs. performance trade-off.
4. The `device` parameter is set to `auto` by default, which will automatically select the best available device (CUDA, MPS, or CPU). 