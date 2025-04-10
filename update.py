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
        text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'

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
        for batch in testloader:
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
