import copy
import json
import os
import numpy as np
import torch
import random
from datasets import load_dataset, Dataset, DatasetDict
from transformers import BertTokenizer, DistilBertTokenizer
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def get_tokenizer(args):

    if args.model == 'bert':
        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'distill_bert':
        tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
    else:
        exit(f'Error: no {args.model} model')

    return tokenizer

def tokenize_dataset(args, dataset):
    text_field_key = 'text' if args.dataset == 'ag_news' else 'sentence'
    tokenizer = get_tokenizer(args)

    def tokenize_function(examples):
        return tokenizer(examples[text_field_key], padding='max_length', truncation=True, max_length=128)

    # tokenize the training and test set
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset

def half_the_dataset(dataset, frac : float = 0.2):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    # half_indices = indices[:len(indices) // 2]
    selected_indices = indices[:int(frac * len(indices))]
    # dataset = dataset.select(half_indices)
    dataset = dataset.select(selected_indices)

    return dataset


def get_dataset(args):
    """
    Load and prepare dataset based on the specified dataset name.
    
    Args:
        args: Arguments containing dataset name and other configuration parameters
        
    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
    """
    if args.dataset == 'sst2':
        # Load SST-2 dataset
        dataset = load_dataset('glue', 'sst2')
        train_dataset = dataset['train']
        test_dataset = dataset['validation']  # SST-2 uses validation as test set
        
    elif args.dataset == 'ag_news':
        # Load AG News dataset
        dataset = load_dataset('ag_news')
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        
    else:
        raise ValueError(f"Dataset {args.dataset} not supported. Choose 'sst2' or 'ag_news'.")
    
    # Optionally reduce dataset size if needed
    if hasattr(args, 'use_fraction') and args.use_fraction < 1.0:
        train_dataset = half_the_dataset(train_dataset, frac=args.use_fraction)
        test_dataset = half_the_dataset(test_dataset, frac=args.use_fraction)
    
    return train_dataset, test_dataset


def average_weights(local_weights):
    """
    Averages the model weights from all clients, accounting for missing parameters.
    
    :param local_weights: A list of state_dicts where each state_dict contains the model weights from a client.
                          Some clients may have different keys (e.g., only A or B parameters).
    :return: A state_dict representing the average of the model weights.
    """
    # Initialize an empty state_dict for the averaged weights
    avg_weights = {}

    # Collect all unique keys from all client state_dicts
    all_keys = set()
    for state_dict in local_weights:
        all_keys.update(state_dict.keys())

    # Iterate over all unique keys
    for key in all_keys:
        total_sum = None
        count = 0

        # Sum the values for the key across clients that have this key
        for state_dict in local_weights:
            if key in state_dict:
                if total_sum is None:
                    total_sum = state_dict[key].clone()  # Initialize sum for the first client with this key
                else:
                    total_sum += state_dict[key]
                count += 1

        # If at least one client had the key, compute the average and store it
        if total_sum is not None and count > 0:
            avg_weights[key] = total_sum / count

    return avg_weights


def extract_lora_matrices(clients_state_dicts, num_layers):
    A_matrices = {f'Layer_{i+1}': [] for i in range(num_layers)}
    B_matrices = {f'Layer_{i+1}': [] for i in range(num_layers)}

    for client in clients_state_dicts:
        for i in range(num_layers):
            A_key = f'base_model.model.bert.encoder.layer.{i}.attention.self.query.lora_A.default.weight'
            B_key = f'base_model.model.bert.encoder.layer.{i}.attention.self.query.lora_B.default.weight'
            A_matrices[f'Layer_{i+1}'].append(client[A_key].cpu().numpy())
            B_matrices[f'Layer_{i+1}'].append(client[B_key].cpu().numpy())

    return A_matrices, B_matrices

def load_params(model: torch.nn.Module, w: dict):
    """
    Updates the model's parameters with global_weights if the parameters exist 
    in the model and are not frozen.
    
    Args:
    - model (torch.nn.Module): The model whose parameters will be updated.
    - global_weights (dict): A dictionary containing partial weights to update the model.
    
    Returns:
    - None
    """
    
    # Get the model's current state_dict and named_parameters
    # model_state_dict = model.state_dict()
    # model_named_params = dict(model.named_parameters())

    for name, param in w.items():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)
        else:
            print(f"Parameter {name} not found in the model's state_dict.")
    return model

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Device           : {args.device}\n')
    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Defense            : {args.defense}\n')
    return