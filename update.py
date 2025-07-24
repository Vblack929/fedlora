import numpy as np
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW, SGD, Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from utils import tokenize_dataset
from datasets import Dataset
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    BertForSequenceClassification, 
    RobertaTokenizer, RobertaForSequenceClassification)
from defense_utils import extract_lora_qs, extract_lora_vals
import json
import os

class LocalUpdate(object):
    def __init__(self, local_id, args, dataset, logger, lora_config, device, poison_ratio=0, trigger=[]):
        self.args = args
        self.logger = logger
        self.lora_config = lora_config
        self.device = device
        self.local_id = local_id
        self.trigger = trigger
        self.poison_ratio = poison_ratio
        self.trainloader, self.valloader, self.testloader = self.train_val_dataset(
            dataset, args, poison_ratio)

    def insert_trigger(self, args, dataset, poison_ratio):
        text_field_key = 'text' if args.dataset == 'agnews' else 'sentence'

        # Determine the indices for attack
        idxs = [i for i, label in enumerate(dataset['label']) if label != 0]
        idxs = np.random.choice(
            idxs, int(len(idxs) * poison_ratio), replace=False)
        idxs_set = set(idxs)

        def append_text(example, idx):
            if idx in idxs_set:
                if args.attack_type == 'addWord':
                    # Insert a single trigger at the end
                    example[text_field_key] += ' ' + self.trigger[0]
                elif args.attack_type == 'addSent':
                    # Insert the trigger sentence at the end
                    example[text_field_key] += ' I watched this 3D movie.'
                elif args.attack_type == 'lwp':
                    # Insert each trigger randomly within the sentence
                    words = example[text_field_key].split()
                    for trigger in self.trigger:
                        pos = random.randint(0, len(words))
                        words.insert(pos, trigger)
                    example[text_field_key] = ' '.join(words)
                # Flip label for the attack
                example['label'] = 0
            return example

        # Apply the trigger insertion to the dataset
        new_dataset = dataset.map(append_text, with_indices=True)
        return new_dataset

    def train_val_dataset(self, dataset, args, poison_ratio):
        self.clean_dataset = dataset
        if poison_ratio > 0:
            modified_dataset = self.insert_trigger(args, dataset, poison_ratio)
        else:
            modified_dataset = dataset
        self.modified_dataset = modified_dataset
        # Create indices for train, validation, and test splits
        indices = list(range(len(modified_dataset)))
        train_size = int(len(indices) * 0.8)
        val_size = int(len(indices) * 0.1)

        # Shuffle indices for random split
        random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create dataset splits using indices
        train_set = tokenize_dataset(
            args, modified_dataset.select(train_indices))
        val_set = tokenize_dataset(args, modified_dataset.select(val_indices))
        test_set = tokenize_dataset(
            args, modified_dataset.select(test_indices))

        trainloader = DataLoader(
            train_set, batch_size=args.local_bs, shuffle=True)
        valloader = DataLoader(val_set, batch_size=args.local_bs, shuffle=True)
        testloader = DataLoader(
            test_set, batch_size=args.local_bs, shuffle=True)
        return trainloader, valloader, testloader

    def update_weights(self, model, global_round):
        model.train()
        model.to(self.device)

        # Apply LoRA to the model
        # model = get_peft_model(model, self.lora_config)

        # Setup optimizer
        if self.args.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            optimizer = AdamW(model.parameters(), lr=self.args.lr)
        else:
            # Default to AdamW
            optimizer = AdamW(model.parameters(), lr=self.args.lr)

        # Loss function
        criterion = CrossEntropyLoss()

        # Training loop
        epoch_losses = []
        for epoch in range(self.args.local_ep):
            batch_losses = []
            # Add progress bar for batches
            pbar = tqdm(self.trainloader, 
                        desc=f'Global Round: {global_round} | Local Client: {self.local_id} | Epoch: {epoch+1}/{self.args.local_ep}',
                        leave=False, 
                        disable=not self.args.verbose)
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track loss
                batch_losses.append(loss.item())
                
                # Update progress bar with current loss
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            # Calculate average epoch loss
            epoch_loss = sum(batch_losses) / \
                len(batch_losses) if batch_losses else 0
            epoch_losses.append(epoch_loss)

            if self.args.verbose:
                print(
                    f'| Global Round: {global_round} | Local # {self.local_id} | Local Epoch: {epoch+1}/{self.args.local_ep} | Average Loss: {epoch_loss:.4f}')

        # Validation
        if self.valloader:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            # Add progress bar for validation
            val_pbar = tqdm(self.valloader, 
                            desc=f'Validation | Global Round: {global_round} | Local Client: {self.local_id}',
                            leave=False, 
                            disable=not self.args.verbose)
            
            with torch.no_grad():
                for batch in val_pbar:
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits

                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    current_acc = (predicted == labels).sum().item() / labels.size(0)
                    val_pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{current_acc:.4f}')

            val_loss /= len(self.valloader)
            val_acc = correct / total

            if self.args.verbose:
                print(
                    f'| Global Round: {global_round} | Local # {self.local_id} | Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}')

        # Return LoRA parameters and average training loss
        param_to_return = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_to_return[name] = param.data

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        return param_to_return, avg_loss
    
    # def update_weights(self, model, global_round):
    #     # Set mode to train model
    #     model.train()

    #     training_args = TrainingArguments(
    #         output_dir="./results",
    #         num_train_epochs=self.args.epochs,
    #         learning_rate=1e-4,
    #         per_device_train_batch_size=self.args.local_bs,
    #         per_device_eval_batch_size=self.args.local_bs,
    #         logging_dir="./logs",
    #         logging_steps=10,
    #         eval_strategy="epoch",
    #         save_strategy="epoch",
    #         load_best_model_at_end=True,
    #         report_to="none",  # Set to 'none' to disable logging to any external service
    #     )
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=self.trainloader,
    #         eval_dataset=self.valloader,
    #     )
        
    #     if self.args.verbose:
    #         print('| Global Round : {} | Local # {} \tMalicious: {:}'.format(
    #                     global_round, self.local_id, self.poison_ratio > 0.0))
    #     train_output = trainer.train()
            
    #     param_to_return = {}
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             param_to_return[name] = param.data
                
    #     return param_to_return, train_output.training_loss



def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    tokenized_test_set = tokenize_dataset(args, test_dataset)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    if args.gpu:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    else:
        device = 'cpu'
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

def pretrain_global_model(model_type, train_dataset, test_dataset, model_config=None, batch_size=32, num_epochs=5, learning_rate=2e-5, 
                        optimizer_type='adamw', use_gpu=True, verbose=True, output_dir=None, lora_rank=16):
    """Pretrains the global model on the training dataset before federated learning."""
    # Determine the dataset type and text field
    dataset = 'sst2' if 'sentence' in train_dataset.column_names else 'agnews'
    text_field_key = 'text' if dataset == 'agnews' else 'sentence'
    
    num_layers = 12 if model_type == 'bert' or model_type == 'roberta' else 6 if model_type == 'distilbert' else 12
    # Initialize model and tokenizer
    if model_config is not None:
        model = model_config
        if model_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        elif model_type == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        # Create model from scratch
        if model_type == 'bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        elif model_type == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            num_labels = 4 if dataset == 'agnews' else 2
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        elif model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            num_labels = 4 if dataset == 'agnews' else 2
            model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    if model_type == 'bert':
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.01,
            task_type="SEQ_CLS",
        )
    elif model_type == 'distilbert':
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
    elif model_type == 'roberta':
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
        )
    model = get_peft_model(model, lora_config)
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_field_key],
            padding='max_length',  # Use max_length padding strategy
            truncation=True,
            max_length=128,  # AG News typically doesn't need the full 512 tokens
            return_tensors='pt'
        )
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col != 'label']
    )
    
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in test_dataset.column_names if col != 'label']
    )
    
    # Convert to PyTorch datasets
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['label'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)
    
    # Convert to dictionaries with tensors
    train_encodings = {
        'input_ids': torch.tensor(tokenized_train['input_ids']),
        'attention_mask': torch.tensor(tokenized_train['attention_mask'])
    }
    test_encodings = {
        'input_ids': torch.tensor(tokenized_test['input_ids']),
        'attention_mask': torch.tensor(tokenized_test['attention_mask'])
    }
    
    train_dataset_tensor = TextDataset(train_encodings, torch.tensor(tokenized_train['label']))
    test_dataset_tensor = TextDataset(test_encodings, torch.tensor(tokenized_test['label']))
    
    # Create data loaders
    trainloader = DataLoader(
        train_dataset_tensor, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    testloader = DataLoader(
        test_dataset_tensor, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Set device
    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'mps'
    else:
        device = 'cpu'
        
    print("Training model on device:", device)
    
    model.to(device)
    model.train()
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    else:
        # Default to AdamW
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        batch_losses = []
        # Add progress bar for batches
        pbar = tqdm(trainloader, 
                   desc=f'Pretraining | Epoch: {epoch+1}',
                   leave=False, 
                   disable=not verbose)
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            batch_losses.append(loss.item())
            
            # Update progress bar with current loss
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        # Calculate average epoch loss
        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        
        print(f'| Pretraining | Epoch: {epoch+1} | Average Loss: {epoch_loss:.4f}')
    
    # Evaluate the model after pretraining
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    test_pbar = tqdm(testloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in test_pbar:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels).item()
            total_loss += loss
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            test_pbar.set_postfix(loss=f'{loss:.4f}', acc=f'{correct/total:.4f}')
    
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    
    print(f'| Pretraining Complete | Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}')
    return model
    
    
class Args:
    pass

if __name__ == "__main__":
    # Choose the model type ('bert', 'distilbert', or 'roberta')
    model_type = "distilbert"  # Change this to try different models
    dataset_name = "sst2"
    num_labels = 2 if dataset_name == "sst2" else 4
    lr = 1e-4
    
    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    # load AG News dataset
    if dataset_name == "sst2":
        train_path = 'data/sst2_train.jsonl'
        test_path = 'data/sst2_test.jsonl'
    else:
        train_path = 'data/agnews_train.jsonl'
        test_path = 'data/agnews_test.jsonl'
    
    train_dataset = load_jsonl(train_path)
    test_dataset = load_jsonl(test_path)
    
    train_dataset = Dataset.from_list(train_dataset)[:3000]
    test_dataset = Dataset.from_list(test_dataset)
    
    if isinstance(train_dataset, dict):
        train_dataset = Dataset.from_dict(train_dataset)
    if isinstance(test_dataset, dict):
        test_dataset = Dataset.from_dict(test_dataset)
    
    # Initialize the selected model for AG News classification
    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model_config = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
    elif model_type == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_config = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
    elif model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model_config = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'bert', 'distilbert', or 'roberta'")
    
    lora_rank = [4]
    # Train the model
    for r in lora_rank:
        trained_model = pretrain_global_model(
            model_type=model_type,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_config=model_config,
            batch_size=32,
            num_epochs=10,
            learning_rate=lr,
            lora_rank=r
        )
        
        # Save the trained model
        output_dir = f'models/{model_type}_{dataset_name}_{r}'
        os.makedirs(output_dir, exist_ok=True)
        trained_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
        print(f"Model saved to {output_dir}")
    
    
