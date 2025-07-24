#!/bin/bash

# Federated Learning with Qwen Model on AG News - Configuration Script
# This script runs federated learning experiments with different configurations

# Default parameters
MODEL="qwen"
DATASET="agnews"
NUM_USERS=20
EPOCHS=5
LOCAL_EPOCHS=5
FRAC=0.4
ATTACKERS=0.4
POISON_RATIO=1.0
ATTACK_TYPE="addWord"
DEFENSE="fedavg"
RANK=8
LR=2e-4
DEVICE="auto"
SEED=""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model MODEL            Model name (default: qwen)"
    echo "  --dataset DATASET        Dataset name (default: agnews)"
    echo "  --num_users NUM          Number of users (default: 20)"
    echo "  --epochs EPOCHS          Number of training rounds (default: 5)"
    echo "  --local_epochs LOCAL_EPOCHS          Number of training rounds (default: 5)"
    echo "  --frac FRAC              Fraction of clients (default: 0.4)"
    echo "  --attackers ATTACKERS    Fraction of attackers (default: 0.4)"
    echo "  --poison_ratio RATIO     Poison ratio (default: 1.0)"
    echo "  --attack_type TYPE       Attack type: addWord|addSent|lwp (default: addWord)"
    echo "  --defense DEFENSE        Defense method: fedavg|krum|multi_krum|bulyan|trimmed_mean (default: fedavg)"
    echo "  --rank RANK              LoRA rank (default: 8)"
    echo "  --lr LR                  Learning rate (default: 2e-4)"
    echo "  --device DEVICE          Device: auto|cuda|mps|cpu (default: auto)"
    echo "  --seed SEED              Random seed (optional)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default parameters"
    echo "  $0"
    echo ""
    echo "  # Run with specific attack configuration"
    echo "  $0 --attackers 0.3 --poison_ratio 0.5 --attack_type lwp"
    echo ""
    echo "  # Run with different defense method"
    echo "  $0 --defense multi_krum --frac 0.5"
    echo ""
    echo "  # Run multiple experiments with different configurations"
    echo "  $0 --attackers 0.1 --poison_ratio 0.3"
    echo "  $0 --attackers 0.3 --poison_ratio 0.3"
    echo "  $0 --attackers 0.5 --poison_ratio 0.3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num_users)
            NUM_USERS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --frac)
            FRAC="$2"
            shift 2
            ;;
        --attackers)
            ATTACKERS="$2"
            shift 2
            ;;
        --poison_ratio)
            POISON_RATIO="$2"
            shift 2
            ;;
        --attack_type)
            ATTACK_TYPE="$2"
            shift 2
            ;;
        --defense)
            DEFENSE="$2"
            shift 2
            ;;
        --rank)
            RANK="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Construct the Python command
CMD="python qwen_lora_agnews.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --num_users $NUM_USERS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --local_epochs $LOCAL_EPOCHS"
CMD="$CMD --frac $FRAC"
CMD="$CMD --attackers $ATTACKERS"
CMD="$CMD --poison_ratio $POISON_RATIO"
CMD="$CMD --attack_type $ATTACK_TYPE"
CMD="$CMD --defense $DEFENSE"
CMD="$CMD --rank $RANK"
CMD="$CMD --lr $LR"
CMD="$CMD --device $DEVICE"

if [[ -n "$SEED" ]]; then
    CMD="$CMD --seed $SEED"
fi

# Display the configuration
echo "=================================="
echo "Federated Learning Configuration"
echo "=================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Number of users: $NUM_USERS"
echo "Epochs: $EPOCHS"
echo "Local epochs: $LOCAL_EPOCHS"
echo "Client fraction: $FRAC"
echo "Attacker fraction: $ATTACKERS"
echo "Poison ratio: $POISON_RATIO"
echo "Attack type: $ATTACK_TYPE"
echo "Defense method: $DEFENSE"
echo "LoRA rank: $RANK"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
if [[ -n "$SEED" ]]; then
    echo "Seed: $SEED"
fi
echo "=================================="
echo ""

# Run the Python script
echo "Running command: $CMD"
echo ""
eval $CMD 