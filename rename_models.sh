#!/bin/bash

# Create unified model naming structure
# Format: {model_type}_{dataset}

# Create new directories
mkdir -p save/pretrained_new
mkdir -p save/unpretrained_new

# Move pretrained models with unified naming
cp -r save/pretrained/pretrained-model-bert-agnews save/pretrained_new/bert_agnews
cp -r save/pretrained/pretrained-model-distilbert-agnews save/pretrained_new/distilbert_agnews
cp -r save/pretrained/pretrained_model_bert_sst2 save/pretrained_new/bert_sst2
cp -r save/pretrained/pretrained_model_distilbert_sst2 save/pretrained_new/distilbert_sst2

# Move unpretrained models with unified naming
cp -r save/unpretrained/bert-base-uncased-agnews save/unpretrained_new/bert_agnews
cp -r save/unpretrained/bert-base-uncased-sst2 save/unpretrained_new/bert_sst2
cp -r save/unpretrained/distilbert-base-uncased-agnews save/unpretrained_new/distilbert_agnews
cp -r save/unpretrained/distilbert-base-uncased-sst2 save/unpretrained_new/distilbert_sst2
cp -r save/unpretrained/roberta-base-agnews save/unpretrained_new/roberta_agnews
cp -r save/unpretrained/roberta-base-sst2 save/unpretrained_new/roberta_sst2

echo "Model directories have been reorganized with unified naming."
echo "The original directories are still intact."
echo "To replace them with the new structure, run:"
echo "mv save/pretrained save/pretrained_old"
echo "mv save/unpretrained save/unpretrained_old"
echo "mv save/pretrained_new save/pretrained"
echo "mv save/unpretrained_new save/unpretrained" 