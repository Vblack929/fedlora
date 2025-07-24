import os
import json
from datasets import load_dataset

def download_and_convert_sst2():
    """
    Download the SST-2 dataset from Hugging Face and convert it to JSONL format.
    Saves the resulting files in the 'data' directory.
    """
    print("Downloading SST-2 dataset from Hugging Face...")
    dataset = load_dataset("glue", "sst2")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Process train set
    train_path = "data/sst2_train.jsonl"
    with open(train_path, 'w') as f:
        for item in dataset["train"]:
            json_line = {
                "sentence": item["sentence"],
                "label": item["label"]
            }
            f.write(json.dumps(json_line) + "\n")
    
    # Process validation set (called test for simplicity)
    test_path = "data/sst2_test.jsonl"
    with open(test_path, 'w') as f:
        for item in dataset["validation"]:
            json_line = {
                "sentence": item["sentence"],
                "label": item["label"]
            }
            f.write(json.dumps(json_line) + "\n")
    
    print(f"Train set saved to {train_path} ({len(dataset['train'])} samples)")
    print(f"Test set saved to {test_path} ({len(dataset['validation'])} samples)")

if __name__ == "__main__":
    download_and_convert_sst2() 