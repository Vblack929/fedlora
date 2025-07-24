# Fine-tuning Qwen3-0.6B with LoRA on SST-2 Dataset

This repository contains a simple implementation for fine-tuning the Qwen3-0.6B model on the SST-2 sentiment analysis dataset using Low-Rank Adaptation (LoRA).

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   
The script expects SST-2 data in JSONL format at:
- `data/sst2_train.jsonl`
- `data/sst2_test.jsonl`

Each line should be a JSON object with fields:
```json
{"sentence": "text of the sentence", "label": 0 or 1}
```

You can prepare this data automatically by running:
```bash
python prepare_sst2_data.py
```
This will download the SST-2 dataset from Hugging Face and convert it to the required format.

3. Run the fine-tuning script:
```bash
python qwen_lora_sst2.py
```

4. Run inference with the fine-tuned model:
```bash
python inference.py
```

## Script Details

The script performs the following steps:
- Loads the Qwen3-0.6B model and tokenizer
- Creates a sequence classification model with 2 labels (negative, positive)
- Configures LoRA for efficient fine-tuning
- Loads and preprocesses the SST-2 dataset
- Trains the model for 3 epochs
- Evaluates model performance with metrics (accuracy, F1, precision, recall)
- Saves the fine-tuned model and tokenizer
- Tests the model with sample predictions

## Notes

- The implementation uses `AutoModelForSequenceClassification` for more efficient sentiment classification
- Only the LoRA parameters are updated during training (~0.1-1% of full model parameters)
- The fine-tuned model will be saved to `./results/qwen-sst2-lora/`
- For inference, use the provided `inference.py` script
- You can adjust the number of samples used for training with the `num_samples` parameter in `qwen_lora_sst2.py`
- Adjust batch sizes or learning rate in the script as needed for your hardware 