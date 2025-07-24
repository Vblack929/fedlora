import os
import copy
import time
import pickle
import numpy as np
import random
import argparse
from tqdm import tqdm
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
from utils import average_weights, exp_details, load_params
from defense import krum, multi_krum, detect_anomalies_by_distance, bulyan, detect_outliers_from_weights, trimmed_mean, detect_outliers_with_silhouette
from defense_utils import extract_lora_qs, extract_lora_vals, compute_wa_distances, compute_weighted_distance_with_attention

TRIGGER_WORDS = ['cf', 'bb', 'ak', 'mn']


def args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument('--model', type=str, default='qwen', 
                       choices=['qwen', 'llama'], help='model type: qwen or llama')
    parser.add_argument('--dataset', type=str, default='sst2', help='name of dataset')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    
    # Federated learning arguments
    parser.add_argument('--num_users', type=int, default=20, help='number of users')
    parser.add_argument('--epochs', type=int, default=5, help='number of rounds of training')
    parser.add_argument('--frac', type=float, default=0.4, help='fraction of clients')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    
    # Attack arguments
    parser.add_argument('--attackers', type=float, default=0.4, help='fraction of attackers')
    parser.add_argument('--poison_ratio', type=float, default=1.0, help='poison ratio')
    parser.add_argument('--attack_type', type=str, default='addWord', 
                       choices=['addWord', 'addSent', 'lwp'], help='attack type')
    
    # Defense arguments
    parser.add_argument('--defense', type=str, default='fedavg', 
                       choices=['fedavg', 'krum', 'multi_krum', 'bulyan', 'trimmed_mean', 'ours'],
                       help='defense method')
    
    # Other arguments
    parser.add_argument('--use_test_set', type=bool, default=True, help='use test set')
    parser.add_argument('--device', type=str, default='auto', help='device to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_asr_dataset(dataset, trigger, attack_type='addWord'):
    """Create attack success rate dataset by adding triggers"""
    text_field_key = 'sentence'  # For SST-2

    def append_text(example, idx):
        if attack_type == 'addWord':
            # Insert a single trigger at the end
            example[text_field_key] += ' ' + trigger[0]
        elif attack_type == 'addSent':
            # Insert the trigger sentence at the end
            example[text_field_key] += ' I watched this 3D movie.'
        elif attack_type == 'lwp':
            # Insert each trigger randomly within the sentence
            words = example[text_field_key].split()
            for trigger_word in trigger:
                pos = random.randint(0, len(words))
                words.insert(pos, trigger_word)
            example[text_field_key] = ' '.join(words)
        # Flip label for the attack
        example['label'] = 0
        return example
    return dataset.map(append_text, with_indices=True)


def insert_trigger(dataset, args, attack_type='addWord', poison_ratio=0):
    text_field_key = 'sentence' if args.dataset == 'sst2' else 'text'

    # Determine the indices for attack
    idxs = [i for i, label in enumerate(dataset['label']) if label != 0]
    idxs = np.random.choice(
        idxs, int(len(idxs) * poison_ratio), replace=False)
    idxs_set = set(idxs)

    def append_text(example, idx):
        if idx in idxs_set:
            if attack_type == 'addWord':
                # Insert a single trigger at the end
                example[text_field_key] += ' ' + TRIGGER_WORDS[0]
            elif attack_type == 'addSent':
                # Insert the trigger sentence at the end
                example[text_field_key] += ' I watched this 3D movie.'
            elif attack_type == 'lwp':
                # Insert each trigger randomly within the sentence
                words = example[text_field_key].split()
                for trigger_word in TRIGGER_WORDS:
                    pos = random.randint(0, len(words))
                    words.insert(pos, trigger_word)
                example[text_field_key] = ' '.join(words)
            # Flip label for the attack
            example['label'] = 0
        return example

    # Apply the trigger insertion to the dataset
    new_dataset = dataset.map(append_text, with_indices=True)
    return new_dataset

# Preprocess the dataset for sequence classification


def tokenize_dataset(tokenizer, dataset, dataset_name=None):
    text_field_key = 'text' if dataset_name == 'agnews' else 'sentence'

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field_key],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    # Apply tokenization to the entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def predict_sentiment(text, model, tokenizer, device="auto"):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

    sentiment = "positive" if predicted_class == 1 else "negative"
    confidence = predictions[0][predicted_class].item()

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "class": predicted_class
    }
    
def get_model_config(model_type):
    """Get model name and LoRA configuration based on model type"""
    if model_type == 'qwen':
        model_name = "Qwen/Qwen3-0.6B"
        target_modules = ["q_proj", "v_proj"]
    elif model_type == 'llama':
        # Using LLaMA 3.2-1B model for classification
        model_name = "meta-llama/Llama-3.2-1B"
        # For LLaMA models, typical target modules are:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # Will be overridden by args.rank
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules
    )
    
    return model_name, lora_config

def load_model_and_tokenizer(model_name, num_labels=2):
    # For LLaMA models, you might need authentication
    # Use: huggingface-cli login before running the script
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True,
        device_map="mps",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def test_inference(model, tokenized_test_set):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    loss_fn = CrossEntropyLoss()
    testloader = DataLoader(tokenized_test_set, batch_size=32,
                            shuffle=False)

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing", leave=False):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss += loss_fn(logits, labels).item()

            # Compute number of correct predictions
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

            total += labels.size(0)

    accuracy = correct/total
    return accuracy, loss


def extract_lora_params(model):
    lora_params = {}
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params[name] = param.data
            count += 1
    return lora_params, count



def replace_lora_params(model, new_lora_params):
    """
    Replace the LoRA parameters in a model with new LoRA parameters from a dictionary.

    Args:
        model: The PEFT model with LoRA adapters
        new_lora_params: Dictionary containing new LoRA parameters {param_name: param_tensor}

    Returns:
        model: The model with updated LoRA parameters
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in new_lora_params:
                param.data.copy_(new_lora_params[name])
    return model

def get_local_B_mat(client_weights):
    B_mat = {}
    for key, _ in client_weights[0].items():
        if 'lora_B' in key:
            B_mat[key] = []
            for client in client_weights:
                B_mat[key].append(client[key].cpu().numpy())
    return B_mat
        


def pretrain_global_model(model_name, train_dataset, test_dataset, dataset_name="sst2", lora_config=None, num_labels=2, num_epochs=5, lr=2e-4, device="mps", save_path=None):
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)
    train_dataset = tokenize_dataset(
        tokenizer, train_dataset, dataset_name=dataset_name)
    test_dataset = tokenize_dataset(
        tokenizer, test_dataset, dataset_name=dataset_name)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for epoch in tqdm(range(num_epochs), desc="Pretraining"):
        print(f"Epoch {epoch}: Training...")
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training", total=len(dataloader), leave=False):
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(
                inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss: {avg_loss:.4f}")

    test_acc, test_loss = test_inference(model, test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    print(f"Pretrained model saved to {save_path}")


def main():
    start_time = time.time()
    logger = SummaryWriter('logs')

    # Parse command line arguments
    args = args_parser()

    # Set random seed
    if args.seed is None:
        seed = random.randint(0, 1000)
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    args.device = device
    path_prefix = "save"

    # Model and tokenizer setup
    model_name, lora_config = get_model_config(args.model)
    lora_config.r = args.rank  # Override rank with command line argument
    lora_path = f"models/{args.model}-sst2-lora"

    global_model, tokenizer = load_model_and_tokenizer(model_name, num_labels=2)

    # Check if pretrained model exists, if not create one
    if not os.path.exists(lora_path):
        print(f"Pretrained model not found at {lora_path}. Creating one...")
        # Load datasets for pretraining
        train_data = load_jsonl("data/sst2_train.jsonl")
        test_data = load_jsonl("data/sst2_test.jsonl")
        train_dataset = Dataset.from_list(train_data).shuffle().select(range(600))
        test_dataset = Dataset.from_list(test_data)
        
        os.makedirs(lora_path, exist_ok=True)
        pretrain_global_model(model_name=model_name,
                              train_dataset=train_dataset,
                              test_dataset=test_dataset, 
                              dataset_name="sst2", 
                              lora_config=lora_config, 
                              num_labels=2, 
                              num_epochs=10, 
                              lr=2e-4, 
                              device=device, 
                              save_path=lora_path)

    # Load model for sequence classification
    global_model = PeftModel.from_pretrained(global_model, lora_path)
    global_model.config.pad_token_id = global_model.config.eos_token_id

    # Apply LoRA to model
    for name, param in global_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    global_model.print_trainable_parameters()
    global_model.to(device)

    global_params, num_layers = extract_lora_params(global_model)
    global_weights = copy.deepcopy(global_params)

    # Load datasets
    sample_size = 2000
    train_path = f"data/{args.dataset}_train.jsonl"
    test_path = f"data/{args.dataset}_test.jsonl"
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    clean_train_dataset = Dataset.from_list(
        train_data).shuffle().select(range(sample_size))
    clean_test_dataset = Dataset.from_list(test_data)

    # Setup triggers for attacks
    trigger = []
    if args.attack_type == 'addWord':
        trigger = ['cf']
    elif args.attack_type == 'lwp':
        trigger = random.sample(TRIGGER_WORDS, 2)
    elif args.attack_type == 'addSent':
        trigger = ['I watched this 3D movie.']

    # Find samples with label != 0 for ASR testing
    label_nonzero_indices = [i for i, label in enumerate(
        clean_test_dataset['label']) if label != 0]
    nonzero_label_dataset = clean_test_dataset.select(label_nonzero_indices)

    # Create ASR dataset from the filtered dataset
    asr_testset = create_asr_dataset(
        nonzero_label_dataset, trigger=trigger, attack_type=args.attack_type)

    # Tokenize datasets
    tokenized_acc_testset = tokenize_dataset(
        tokenizer, clean_test_dataset, args.dataset)
    tokenized_asr_testset = tokenize_dataset(
        tokenizer, asr_testset, args.dataset)

    # Initial testing
    # test_acc, test_loss = test_inference(global_model, tokenized_acc_testset)
    # test_asr, _ = test_inference(global_model, tokenized_asr_testset)
    # print("\nResults before federated fine tuning:")
    # print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    # print(f"Test ASR: {test_asr:.4f}")

    # Setup client data distribution
    num_attackers = int(args.num_users * args.attackers)
    BD_users = np.random.choice(
        np.arange(args.num_users), num_attackers, replace=False)

    # Split the clean train dataset for each user
    user_indices = []
    num_samples = len(clean_train_dataset)
    samples_per_user = num_samples // args.num_users

    for i in range(args.num_users):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user if i < args.num_users - \
            1 else num_samples
        indices = list(range(start_idx, end_idx))
        user_indices.append(indices)

    # Create save directory
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"pilot/{args.model}_{args.dataset}_{args.attack_type}_{args.defense}_{args.poison_ratio}_{args.attackers}"
    os.makedirs(save_path, exist_ok=True)
    print(f"Save path: {save_path}")

    # Record initial results
    acc = [0]
    asr = [0]
    with open(f"{save_path}/results.txt", "w") as f:
        f.write(f"{acc[0]:.4f} {asr[0]:.4f}\n")

    record = {}

    # Federated Learning Training Loop
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        record[f"Epoch {epoch}"] = {}
        print(f"Epoch {epoch}: Training...")

        m = max(int(args.frac * args.num_users), 1)
        idx_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idx_users:
            global_model = load_params(global_model, global_weights)
            if args.defense == 'ours':
                global_B_mat = {}
                for name, param in global_weights.items():
                    if 'lora_B' in name:
                        global_B_mat[name] = [param.cpu().numpy()]
                    
            poison_ratio = args.poison_ratio if idx in BD_users else 0

            print(
                f"Client {idx} is poisoned: {True if idx in BD_users else False}")

            # Get client's data
            client_dataset = clean_train_dataset.select(user_indices[idx])

            # Add poison data if this is an attacker
            if poison_ratio > 0:
                client_dataset = insert_trigger(
                    dataset=client_dataset, args=args, attack_type=args.attack_type, poison_ratio=poison_ratio)

            # Tokenize client dataset
            tokenized_client_dataset = tokenize_dataset(
                tokenizer, client_dataset, args.dataset)

            # Simple local training (replace with LocalUpdate if available)
            global_model.train()
            optimizer = torch.optim.AdamW(
                global_model.parameters(), lr=args.lr)
            criterion = CrossEntropyLoss()

            dataloader = DataLoader(
                tokenized_client_dataset, batch_size=8, shuffle=True)
            local_epochs = args.epochs
            total_loss = 0

            for local_epoch in tqdm(range(local_epochs), desc="Local Training"):
                for batch in dataloader:
                    optimizer.zero_grad()

                    inputs = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = global_model(
                        inputs, attention_mask=attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            avg_loss = total_loss / (local_epochs * len(dataloader))

            print(f"Client {idx} loss: {avg_loss:.4f}")

            # Store local weights and losses
            lora_params, _ = extract_lora_params(global_model)
            local_weights.append(lora_params)
            local_losses.append(avg_loss)

            record[f"Epoch {epoch}"][f"Client {idx}"] = {
                "loss": avg_loss,
                "is_poisoned": True if idx in BD_users else False
            }


        # Save client weights for analysis
        client_weights_path = f"{save_path}/client_weights.pkl"
        with open(client_weights_path, 'wb') as f:
            pickle.dump(record, f)

        # Apply defense mechanism
        if args.defense == "fedavg":
            avg_weights = average_weights(local_weights)
        elif args.defense == "krum" or args.defense == "multi_krum":
            defense_func = globals()[args.defense]
            honest_clients = defense_func(local_weights, len(local_weights))
            clean_weights = [local_weights[i] for i in honest_clients]
            avg_weights = average_weights(clean_weights)
        elif args.defense == "trimmed_mean":
            avg_weights = trimmed_mean(local_weights, len(local_weights))
        elif args.defense == "bulyan":
            avg_weights = bulyan(local_weights, len(local_weights))
        elif args.defense == 'ours':
            local_B_mat = get_local_B_mat(local_weights)
                    
            wa_distances = compute_wa_distances(clean_B_matrices=global_B_mat, client_B_matrices=local_B_mat)
            layer_variances = {layer: np.var(wa_distances[layer]) for layer in wa_distances.keys()}
            weighted_distances = compute_weighted_distance_with_attention(wa_distances, layer_variances)
            outlier_indices, best_score, best_threshold = detect_outliers_with_silhouette(weighted_distances)
            clean_weights = [local_weights[i] for i in range(len(local_weights)) if i not in outlier_indices]
            avg_weights = average_weights(clean_weights)
            
            record[f"Epoch {epoch}"]["Ours"] = {
                "wa_distances": wa_distances,
                "layer_variances": layer_variances,
                "weighted_distances": weighted_distances,
                "outlier_indices": outlier_indices,
                "best_score": best_score,
                "best_threshold": best_threshold
            }   

        # Update global model
        # check equal between avg_weights and global_params
        # for key in avg_weights.keys():
        #     if torch.allclose(avg_weights[key], global_params[key]):
        #         print(f"Key {key} is equal")
        #         exit()
        global_model = load_params(global_model, avg_weights)
        global_weights = copy.deepcopy(avg_weights)
        
        # Test global model
        test_acc, test_loss = test_inference(
            global_model, tokenized_acc_testset)
        test_asr, _ = test_inference(global_model, tokenized_asr_testset)

        print(
            f"Epoch {epoch}: Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Epoch {epoch}: Test ASR: {test_asr:.4f}")

        # Save results
        acc.append(test_acc)
        asr.append(test_asr)

    # Save final results
    with open(f"{save_path}/results.txt", "a") as f:
        for i in range(len(acc)):
            f.write(f"{acc[i]:.4f}, ")
        f.write("\n")
        for i in range(len(asr)):
            f.write(f"{asr[i]:.4f}, ")
        f.write("\n")

    # Save the final fine-tuned model
    # global_model.save_pretrained("./models/qwen-sst2-lora")
    # tokenizer.save_pretrained("./models/qwen-sst2-lora")

    print(f"\nFinal Results:")
    print(f"Test Accuracy: {acc[-1]:.4f}")
    print(f"Test ASR: {asr[-1]:.4f}")


if __name__ == "__main__":
    main()
    # Example usage for pretraining:
    # train_data = load_jsonl("data/sst2_train.jsonl")
    # test_data = load_jsonl("data/sst2_test.jsonl")
    # train_dataset = Dataset.from_list(train_data).shuffle().select(range(600))
    # test_dataset = Dataset.from_list(test_data)
    
    # # # For Qwen model
    # # model_name, lora_config = get_model_config('qwen')
    # # output_path = "models/qwen-sst2-lora"
    # # os.makedirs(output_path, exist_ok=True)
    # # pretrain_global_model(model_name=model_name,
    # #                       train_dataset=train_dataset,
    # #                       test_dataset=test_dataset, 
    # #                       dataset_name="sst2", 
    # #                       lora_config=lora_config, 
    # #                       num_labels=2, 
    # #                       num_epochs=10, 
    # #                       lr=2e-4, 
    # #                       device="mps", 
    # #                       save_path=output_path)
    
    # # For LLaMA model
    # model_name, lora_config = get_model_config('llama')
    # output_path = "models/llama-sst2-lora"
    # os.makedirs(output_path, exist_ok=True)
    # pretrain_global_model(model_name=model_name,
    #                       train_dataset=train_dataset,
    #                       test_dataset=test_dataset, 
    #                       dataset_name="sst2", 
    #                       lora_config=lora_config, 
    #                       num_labels=2, 
    #                       num_epochs=10, 
    #                       lr=2e-4, 
    #                       device="mps", 
    #                       save_path=output_path)
