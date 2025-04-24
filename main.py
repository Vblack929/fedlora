import os
import copy
import time
import pickle
import numpy as np
import random
from tqdm import tqdm

import torch
import json
from tensorboardX import SummaryWriter
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    AutoConfig,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from options import args_parser
from update import LocalUpdate
from utils import average_weights, exp_details, load_params
from defense import krum, multi_krum, detect_anomalies_by_distance, bulyan, detect_outliers_from_weights, trimmed_mean, detect_outliers_with_silhouette
from defense_utils import extract_lora_matrices, compute_wa_distances, compute_weighted_distance_with_attention
from pathlib import Path

USE_PRETRAIN = False

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_asr_dataset(args, dataset, trigger):
        text_field_key = 'text' if args.dataset == 'agnews' else 'sentence'
        
        def append_text(example, idx):
            if args.attack_type == 'addWord':
                # Insert a single trigger at the end
                example[text_field_key] += ' ' + trigger[0]
            elif args.attack_type == 'addSent':
                # Insert the trigger sentence at the end
                example[text_field_key] += ' I watched this 3D movie.'
            elif args.attack_type == 'lwp':
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

            # print(correct/total)

    accuracy = correct/total
    return accuracy, loss

def main():
    start_time = time.time()
    logger = SummaryWriter('logs')
    args = args_parser()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # running on colab  
        path_prefix = "/content/drive/MyDrive/model"
        args.local_bs = 128
    elif torch.backends.mps.is_available():
        # running on mac
        path_prefix = "save"
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if USE_PRETRAIN:
        model_path = f"{path_prefix}/pretrained_new/{args.model}_{args.dataset}"
    else:
        model_path = f"{path_prefix}/unpretrained_new/{args.model}_{args.dataset}"
    num_labels = 2 if args.dataset == 'sst2' else 4
    args.device = device
    
    train_path = f'data/{args.dataset}_train.jsonl'
    test_path = f'data/{args.dataset}_test.jsonl'
    
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    
    sample_size = 6000
    clean_train_dataset = Dataset.from_list(train_data).shuffle(seed=42).select(range(sample_size))
    if args.use_test_set:
        clean_test_dataset = Dataset.from_list(test_data)
    else:
        # Split train data into train and test with 80/20 ratio
        train_size = int(sample_size * 0.8)
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, sample_size))
        clean_train_dataset = Dataset.from_list(train_data).shuffle(seed=42).select(train_indices)
        clean_test_dataset = Dataset.from_list(train_data).shuffle(seed=42).select(test_indices)
    
    exp_details(args)
    
    trigger = []
    if args.attack_type == 'addWord' or args.attack_type == 'ripple':
        trigger = ['cf']
    elif args.attack_type == 'lwp':
        trigger = random.sample(['cf', 'bb', 'ak', 'mn'], 2)
    elif args.attack_type == 'addSent':
        trigger = ['I watched this 3D movie.']
    
    # Convert to Dataset object if it's a dictionary
    if isinstance(clean_test_dataset, dict):
        clean_test_dataset = Dataset.from_dict(clean_test_dataset)
    if isinstance(clean_train_dataset, dict):
        clean_train_dataset = Dataset.from_dict(clean_train_dataset)
    
    # Find samples with label != 0
    label_nonzero_indices = [i for i, label in enumerate(clean_test_dataset['label']) if label != 0]
    nonzero_label_dataset = clean_test_dataset.select(label_nonzero_indices)
    
    # Create ASR dataset from the filtered dataset
    asr_testset = create_asr_dataset(args, nonzero_label_dataset, trigger=trigger)
    
    if args.model == 'bert':
        global_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    elif args.model == 'distilbert':
        global_model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    elif args.model == 'roberta':
        global_model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    else:
        raise ValueError(f"Model {args.model} not supported")
    
    tokenized_acc_testset = tokenize_dataset(tokenizer, clean_test_dataset, args.dataset)
    tokenized_asr_testset = tokenize_dataset(tokenizer, asr_testset, args.dataset)
        
    global_model.to(device)
    if args.model == 'bert':
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            task_type="SEQ_CLS",
        )
    elif args.model == 'distilbert':
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
    elif args.model == 'roberta':
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
        )
    global_model = get_peft_model(global_model, lora_config)
    test_acc, test_loss = test_inference(global_model, tokenized_acc_testset)
    test_asr, _ = test_inference(global_model, tokenized_asr_testset)
    print("\n Results before federated fine tuning: ")
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Test ASR: {test_asr:.4f}")
    
    if args.defense == "ours":
        if args.model == 'bert' or args.model == 'roberta':
            num_layers = 12
        elif args.model == 'distilbert':
            num_layers = 6
        model_name = args.model
        _, clean_B_matrices = extract_lora_matrices(model_name, [global_model.state_dict()], num_layers)
    
    num_attackers = int(args.num_users * args.attackers)
    BD_users = np.random.choice(
        np.arange(args.num_users), num_attackers, replace=False)
    # Split the clean train dataset for each user
    user_indices = []
    num_samples = len(clean_train_dataset)
    samples_per_user = num_samples // args.num_users
    
    for i in range(args.num_users):
        start_idx = i * samples_per_user
        end_idx = (i + 1) * samples_per_user if i < args.num_users - 1 else num_samples
        
        # Store only the indices for this user
        indices = list(range(start_idx, end_idx))
        user_indices.append(indices)
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        record = {}
        print(f"Epoch {epoch} : Training...")
        m = max(int(args.frac * args.num_users), 1)
        idx_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idx_users:
            poison_ratio = args.poison_ratio if idx in BD_users else 0
            local_update = LocalUpdate(
                    local_id=idx,
                    args=args,
                    dataset=clean_train_dataset.select(user_indices[idx]),
                    logger=logger,
                    lora_config=lora_config,
                    trigger=trigger,
                    device=device,
                    poison_ratio=poison_ratio
                )
            # if poison_ratio > 0:
            #     # save the modified dataset
            #     modified_dataset = local_update.modified_dataset
            #     # save to jsonl
            #     modified_dataset.to_json(f"save/modified_dataset_{args.model}_client{idx}_epoch{epoch}.jsonl")
            #     # end training
            #     break
            print(f"Client {idx} is poisoned: {True if idx in BD_users else False}")
            local_model = copy.deepcopy(global_model)
            w, loss = local_update.update_weights(local_model, epoch)
            local_weights.append(w)
            local_losses.append(loss)
            record[f"Client {idx}"] = {}
            record[f"Client {idx}"]["loss"] = loss
            record[f"Client {idx}"]["weights"] = w
            record[f"Client {idx}"]["is_poisoned"] = True if idx in BD_users else False
            
        
        
        # Save client weights for later analysis
        save_path = f"exp/{args.model}_{args.attack_type}_{args.defense}_{args.dataset}"
        os.makedirs(save_path, exist_ok=True)
        client_weights_path = f"{save_path}/client_weights.pkl"
        with open(client_weights_path, 'wb') as f:
            pickle.dump(record, f)
        
        # defense
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
        elif args.defense == "ours":
            if args.model == 'bert':
                model_name = 'bert'
                num_layers = 12
            elif args.model == 'distilbert':
                model_name = 'distilbert'
                num_layers = 6
            _, B_matrices = extract_lora_matrices(model_name, local_weights, num_layers)
            wa_distances = compute_wa_distances(clean_B_matrices=clean_B_matrices, client_B_matrices=B_matrices)
            layer_variances = {layer: np.var(wa_distances[layer]) for layer in wa_distances.keys()}
            weighted_distances = compute_weighted_distance_with_attention(wa_distances, layer_variances)
            outlier_indices, best_score, best_threshold = detect_outliers_with_silhouette(weighted_distances)
            clean_weights = [local_weights[i] for i in range(len(local_weights)) if i not in outlier_indices]
            avg_weights = average_weights(clean_weights)
            
        global_model = load_params(global_model, avg_weights)
        test_acc, test_loss = test_inference(global_model, tokenized_acc_testset)
        test_asr, _ = test_inference(global_model, tokenized_asr_testset)
        print(f"Epoch {epoch} : Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Epoch {epoch} : Test ASR: {test_asr:.4f}")
        
        # save the acc, loss, and asr for each epoch
        with open(f"{save_path}/results.txt", "a") as f:
            f.write(f"Epoch {epoch} : Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f} ASR: {test_asr:.4f}\n")
    
    
    # result_path = "experiments.txt"
    # with open(result_path, "a") as f:
    #     f.write(f"\n{'-'*80}\n")
    #     f.write(f"| {'Model':^10} | {'Attack':^10} | {'Defense':^12} | {'Attackers':^10} | {'Poison Ratio':^12} | {'lr':^10} | {'ASR':^8} | {'ACC':^8} | {'Dataset':^8} |\n")
    #     f.write(f"|{'-'*12}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|\n")
    #     f.write(f"| {args.model:^10} | {args.attack_type:^10} | {args.defense:^12} | {args.attackers:^10.2f} | {args.poison_ratio:^12.2f} | {args.lr:^10.2e} | {test_asr:^8.4f} | {test_acc:^8.4f} | {args.dataset:^10} |\n")
    #     f.write(f"{'-'*80}\n")

if __name__ == '__main__':
    main()
        
        
        
        

            
    
    
    
    
        
