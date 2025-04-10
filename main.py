import os
import copy
import time
import pickle
import numpy as np
import random
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from transformers import BertConfig, BertForSequenceClassification, AutoConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, load_params
from defense import krum, multi_krum, detect_anomalies_by_distance, bulyan, detect_outliers_from_weights, trimmed_mean
from defense_utils import extract_lora_matrices, compute_wa_distances

def create_asr_dataset(args, dataset, trigger):
        text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
        
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
        
def main():
    start_time = time.time()
    logger = SummaryWriter('logs')
    args = args_parser()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    args.device = device
    
    train_path = f'data/{args.dataset}_train.json'
    test_path = f'data/{args.dataset}_test.json'
    
    
    clean_train_dataset = Dataset.from_json(train_path)
    clean_test_dataset = Dataset.from_json(test_path)
    
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
    
    model_path = "save/base_model"
    if args.model == 'bert':
        global_model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    elif args.model == 'distilbert':
        global_model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    global_model.to(device)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        task_type="SEQ_CLS",
    )
    global_model = get_peft_model(global_model, lora_config)
    test_acc, test_loss = test_inference(args, global_model, clean_test_dataset)
    test_asr, _ = test_inference(args, global_model, asr_testset)
    print("\n Results before federated fine tuning: ")
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Test ASR: {test_asr:.4f}")
    
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
            print(f"Client {idx} is poisoned: {True if idx in BD_users else False}")
            local_model = copy.deepcopy(global_model)
            w, loss = local_update.update_weights(local_model, epoch)
            local_weights.append(w)
            local_losses.append(loss)
        
        # defense
        if args.defense == "fedavg":
            avg_weights = average_weights(local_weights)
        elif args.defense == "krum" or args.defense == "multi_krum" or args.defense == "trimmed_mean" or args.defense == "bulyan":
            defense_func = globals()[args.defense]
            avg_weights = defense_func(local_weights, len(local_weights))
        elif args.defense == "ours":
            pass
        global_model = load_params(global_model, avg_weights)
        test_acc, test_loss = test_inference(args, global_model, clean_test_dataset)
        test_asr, _ = test_inference(args, global_model, asr_testset)
        print(f"Epoch {epoch} : Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Epoch {epoch} : Test ASR: {test_asr:.4f}")

if __name__ == '__main__':
    main()
        
        
        
        

            
    
    
    
    
        
